import supervisely as sly
import src.globals as g
from typing import Literal
from supervisely.app.widgets import Progress
from supervisely.api.module_api import ApiField
from supervisely.annotation.label import LabelJsonFields
from supervisely.annotation.annotation import AnnotationJsonFields


project_progress_bar = Progress(hide_on_finish=True)
images_update_bar = Progress(hide_on_finish=True)


def inference_json_anno_preprocessing(
    ann, temp_meta: sly.ProjectMeta, suffix: Literal["bbox", "mask"] = None
) -> sly.Annotation:
    for i, obj in enumerate(ann["annotation"]["objects"]):
        class_: str = obj["classTitle"]
        ann["annotation"]["objects"][i]["classTitle"] = class_
        if suffix == "bbox":
            class_type = sly.Rectangle
        elif suffix == "mask":
            class_type = sly.Bitmap
        else:
            raise NotImplementedError("Suffix should be either 'bbox' or 'mask'")
        if temp_meta.get_obj_class(class_) is None:
            class_color = None
            orig_class = temp_meta.get_obj_class(class_.rstrip(f"_{suffix}"))
            if orig_class is None and class_.endswith("_mask"):
                orig_class = temp_meta.get_obj_class(class_[:-4] + "bbox")
            if orig_class is not None:
                class_color = orig_class.color
            new_obj_class = sly.ObjClass(class_, class_type, class_color)
            temp_meta = temp_meta.add_obj_class(new_obj_class)
    if temp_meta.get_tag_meta("confidence") is None:
        temp_meta = temp_meta.add_tag_meta(sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER))
    return sly.Annotation.from_json(ann["annotation"], temp_meta), temp_meta


def apply_to_project_event(
    destination_project: sly.app.widgets.DestinationProject,
    inference_settings: dict,
    F_MODEL_DATA: dict,
    S_MODEL_DATA: dict,
):
    if F_MODEL_DATA.get("session_id") is None or S_MODEL_DATA.get("session_id") is None:
        sly.app.show_dialog(
            "Warning",
            f"Please connect to both models before applying them to the project",
            status="warning",
        )
        sly.logger.warning("Please connect to both models before applying them to the project")
        return

    def update_proj_meta_classes(
        ann: dict,
        output_project_meta: sly.ProjectMeta,
        suffix: Literal["bbox", "mask"],
    ) -> sly.ProjectMeta:
        project_meta_needs_update = False
        for label in ann[ApiField.ANNOTATION][AnnotationJsonFields.LABELS]:
            new_class_: str = label[LabelJsonFields.OBJ_CLASS_NAME]
            if output_project_meta.get_obj_class(new_class_) is None:
                sly.logger.debug(f"Adding {new_class_} to the project meta")
                if suffix == "bbox":
                    geometry_type = sly.Rectangle
                elif suffix == "mask":
                    geometry_type = sly.Bitmap
                else:
                    raise NotImplementedError("Suffix should be either 'bbox' or 'mask'")
                class_color = None
                orig_class = output_project_meta.get_obj_class(new_class_.rstrip(f"_{suffix}"))
                if orig_class is None and new_class_.endswith("_mask"):
                    orig_class = output_project_meta.get_obj_class(new_class_[:-4] + "bbox")
                if orig_class is not None:
                    class_color = orig_class.color
                new_obj_class = sly.ObjClass(new_class_, geometry_type, color=class_color)
                output_project_meta = output_project_meta.add_obj_class(new_obj_class)
                project_meta_needs_update = True

        if project_meta_needs_update:
            g.api.project.update_meta(output_project_id, output_project_meta)
            sly.logger.debug(f"Project meta successfully updated")
        return output_project_meta

    output_project_id = destination_project.get_selected_project_id()
    if output_project_id is None:
        output_project_name = destination_project.get_project_name()
        if output_project_name.strip() == "":
            output_project_name = f"{g.project_info.name} - (Annotated)"
        output_project = g.api.project.create(
            workspace_id=g.workspace.id,
            name=output_project_name,
            type=sly.ProjectType.IMAGES,
            change_name_if_conflict=True,
        )
        output_project_id = output_project.id

    def get_output_ds(destination_project: sly.app.widgets.DestinationProject, dataset_name):
        use_project_datasets_structure = destination_project.use_project_datasets_structure()
        if use_project_datasets_structure is True:
            output_dataset_name = dataset_name
        else:
            output_dataset_id = destination_project.get_selected_dataset_id()
            if not output_dataset_id:
                output_dataset_name = destination_project.get_dataset_name()
                if not output_dataset_name or output_dataset_name.strip() == "":
                    output_dataset_name = "ds"
            else:
                output_dataset_info = g.api.dataset.get_info_by_id(output_dataset_id)
                output_dataset_name = output_dataset_info.name
        output_dataset = g.api.dataset.get_or_create(output_project_id, output_dataset_name)

        return output_dataset.id

    # merge project metas and add tag "confidence" to it
    output_project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(output_project_id))
    output_project_meta = output_project_meta.merge(g.project_meta)
    if output_project_meta.get_tag_meta("confidence") is None:
        output_project_meta = output_project_meta.add_tag_meta(
            sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER)
        )
    g.api.project.update_meta(output_project_id, output_project_meta)

    # apply models to project
    datasets_list = [g.api.dataset.get_info_by_id(ds_id) for ds_id in g.DATASET_IDS]
    total_items_cnt = sum([ds.items_count for ds in datasets_list])
    with project_progress_bar(
        message="Uploading new items to project...", total=total_items_cnt
    ) as pbar:
        for dataset in datasets_list:
            images_infos = g.api.image.get_list(dataset.id)
            output_dataset_id = get_output_ds(destination_project, dataset.name)
            result_anns = []
            with images_update_bar(
                message=f"Processing {dataset.name} dataset...", total=len(images_infos)
            ) as i_pbar:
                for image in images_infos:
                    temp_inference_settings = inference_settings.copy()
                    new_anns_objects_added = 0
                    # get existing and new annotations
                    image_ann = g.api.annotation.download(image.id)
                    image_ann = sly.Annotation.from_json(image_ann.annotation, g.project_meta)
                    sly.logger.debug(
                        f"Sending request to generate predictions for {image.id} image..."
                    )
                    f_new_ann = g.api.task.send_request(
                        F_MODEL_DATA["session_id"],
                        "inference_image_id",
                        data={"image_id": image.id, "settings": temp_inference_settings},
                        timeout=500,
                    )
                    f_image_ann = None
                    if g.save_bboxes:
                        output_project_meta = update_proj_meta_classes(
                            f_new_ann, output_project_meta, suffix="bbox"
                        )
                        f_image_ann = sly.Annotation.from_json(
                            f_new_ann[ApiField.ANNOTATION], output_project_meta
                        )
                        new_anns_objects_added += len(
                            f_new_ann[ApiField.ANNOTATION][AnnotationJsonFields.LABELS]
                        )
                    s_labels = []
                    for label in f_new_ann[ApiField.ANNOTATION][AnnotationJsonFields.LABELS]:
                        class_name = label["classTitle"].rstrip("_bbox")
                        rectangle = {"points": label["points"]}
                        temp_inference_settings.update(
                            {
                                "input_image_id": image.id,
                                "mode": "bbox",
                                "rectangle": rectangle,
                                "bbox_class_name": class_name,
                            }
                        )
                        s_new_ann = g.api.task.send_request(
                            S_MODEL_DATA["session_id"],
                            "inference_image_id",
                            data={"image_id": image.id, "settings": temp_inference_settings},
                            timeout=500,
                        )
                        s_labels.extend(s_new_ann[ApiField.ANNOTATION][AnnotationJsonFields.LABELS])
                    s_new_ann[ApiField.ANNOTATION][AnnotationJsonFields.LABELS] = s_labels
                    sly.logger.debug(f"Updating the project meta with new classes")
                    output_project_meta = update_proj_meta_classes(
                        s_new_ann, output_project_meta, suffix="mask"
                    )
                    new_anns_objects_added += len(
                        s_new_ann[ApiField.ANNOTATION][AnnotationJsonFields.LABELS]
                    )
                    sly.logger.debug(f"Merging new and existing annotations")

                    s_image_ann = sly.Annotation.from_json(
                        s_new_ann[ApiField.ANNOTATION], output_project_meta
                    )

                    result_ann = image_ann.add_labels(s_image_ann.labels)
                    if f_image_ann is not None:
                        result_ann = result_ann.add_labels(f_image_ann.labels)
                    result_anns.append(result_ann)
                    if new_anns_objects_added > 0:
                        sly.logger.debug(
                            f"New annotations added to image: {new_anns_objects_added}"
                        )
                    else:
                        sly.logger.info(f"No objects were added during inference for this batch")
                    i_pbar.update(1)
            image_names = [image_info.name for image_info in images_infos]
            img_ids = [image_info.id for image_info in images_infos]
            sly.logger.debug(f"Uploading {len(image_names)} images")
            uploaded_images_infos = g.api.image.upload_ids(
                output_dataset_id,
                image_names,
                img_ids,
                conflict_resolution=destination_project.get_conflict_resolution(),
            )
            img_ids = [image_info.id for image_info in uploaded_images_infos]
            sly.logger.debug(f"Uploading {len(result_anns)} annotations")
            g.api.annotation.upload_anns(img_ids, result_anns)

            pbar.update(len(img_ids))
    g.api.app.workflow.add_input_project(g.project_info.id)
    g.api.app.workflow.add_output_project(output_project_id)
    return g.api.project.get_info_by_id(output_project_id)
