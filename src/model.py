import supervisely as sly
from typing import List, Literal


def inference_json_anno_preprocessing(
    ann, temp_meta: sly.ProjectMeta, suffix: Literal["bbox", "mask"] = None
) -> sly.Annotation:
    # temp_meta = project_meta.clone()
    # pred_classes = []
    for i, obj in enumerate(ann["annotation"]["objects"]):
        class_: str = obj["classTitle"]
        # pred_classes.append(class_)
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
            if orig_class is not None:
                class_color = orig_class.color
            new_obj_class = sly.ObjClass(class_, class_type, class_color)
            temp_meta = temp_meta.add_obj_class(new_obj_class)
    if temp_meta.get_tag_meta("confidence") is None:
        temp_meta = temp_meta.add_tag_meta(sly.TagMeta("confidence", sly.TagValueType.ANY_NUMBER))
    return sly.Annotation.from_json(ann["annotation"], temp_meta), temp_meta
