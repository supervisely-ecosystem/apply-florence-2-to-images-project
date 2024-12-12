import random
from typing import List

from src.utils import apply_to_project_event, project_progress_bar, images_update_bar
from src.ui.classes_mapping_prompts import ClassesMappingWithPrompts

import supervisely as sly
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    Stepper,
    ProjectThumbnail,
    GridGallery,
    InputNumber,
    Field,
    Progress,
    SelectAppSession,
    DoneLabel,
    ModelInfo,
    SelectDataset,
    DestinationProject,
    Flexbox,
    Grid,
    Empty,
    Text,
    Collapse,
    Switch,
)

import src.globals as g
from src.model import inference_json_anno_preprocessing

IS_IMAGE_PROMPT = True
PREVIEW_IMAGES_INFOS = []
CURRENT_REF_IMAGE_INDEX = 0
REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]
F_MODEL_DATA = {}
S_MODEL_DATA = {}


# fetching some images for preview
def get_images_infos_for_preview():
    if len(g.DATASET_IDS) > 0:
        datasets_list = [g.api.dataset.get_info_by_id(ds_id) for ds_id in g.DATASET_IDS]
    else:
        datasets_list = g.api.dataset.get_list(g.project_id)
    IMAGES_INFO_LIST = []
    for dataset in datasets_list:
        if len(datasets_list) == 1:
            samples_count = dataset.images_count
        else:
            samples_count = dataset.images_count * (100 - len(datasets_list)) // 100
        if samples_count == 0:
            break

        IMAGES_INFO_LIST += random.sample(g.api.image.get_list(dataset.id), samples_count)
        if len(IMAGES_INFO_LIST) >= 1000:
            break
    return IMAGES_INFO_LIST


IMAGES_INFO_LIST = get_images_infos_for_preview()


# ------------------------------------- Input Data Selection ------------------------------------- #

dataset_selector = SelectDataset(
    project_id=g.project_id, multiselect=True, select_all_datasets=True
)


def on_dataset_selected(new_dataset_ids=None):
    global IMAGES_INFO_LIST, CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY

    if not new_dataset_ids:
        new_dataset_ids = dataset_selector.get_selected_ids()

    new_project_id = dataset_selector._project_selector.get_selected_id()
    if new_project_id != g.project_id:
        dataset_selector.disable()
        g.project_id = new_project_id
        g.project_info = g.api.project.get_info_by_id(g.project_id)
        g.project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(g.project_id))
        g.workspace = g.api.workspace.get_info_by_id(g.project_info.workspace_id)
        dataset_selector.enable()
        g.DATASET_IDS = new_dataset_ids
        IMAGES_INFO_LIST = get_images_infos_for_preview()
        CURRENT_REF_IMAGE_INDEX = 0
        REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]
    else:
        if set(g.DATASET_IDS) != set(new_dataset_ids):
            g.DATASET_IDS = new_dataset_ids
            IMAGES_INFO_LIST = get_images_infos_for_preview()
            CURRENT_REF_IMAGE_INDEX = 0
            REF_IMAGE_HISTORY = [CURRENT_REF_IMAGE_INDEX]

    sly.logger.info(
        f"Team: {g.team.id} \t Project: {g.project_info.id} \t Datasets: {g.DATASET_IDS}"
    )

    if len(new_dataset_ids) == 0:
        button_download_data.disable()
    else:
        button_download_data.enable()


dataset_selector.value_changed(on_dataset_selected)
button_download_data = Button("Select data")


@button_download_data.click
def download_data():
    on_dataset_selected()

    if data_card.is_disabled() is True:
        toggle_cards(["data_card"], enabled=True)
        toggle_cards(
            [
                "select_models_card",
                "classess_settings_card",
                "preview_card",
                "apply_to_project_card",
            ],
            enabled=False,
        )
        button_download_data.enable()
        button_download_data.text = "Select data"
        florence_set_model_type_button.disable()
        florence_set_model_type_button.text = "Select model"
        sam2_set_model_type_button.disable()
        sam2_set_model_type_button.text = "Select model"
        set_classess_prompts_button.disable()
        set_classess_prompts_button.text = "Set settings"
        new_random_images_preview_button.disable()
        get_predictions_preview_button.disable()
        input_project_thmb.hide()
        apply_to_project_button.disable()
        select_models_card.collapse()
        classess_settings_card.collapse()
        preview_card.collapse()
        apply_to_project_card.collapse()
        stepper.set_active_step(1)
    else:
        button_download_data.disable()
        toggle_cards(
            [
                "data_card",
                "inference_type_selection_card",
                "classess_settings_card",
                "preview_card",
                "apply_to_project_card",
            ],
            enabled=False,
        )

        input_project_thmb.set(info=g.project_info)
        input_project_thmb.show()
        classes_mapping.set(g.project_meta.obj_classes)
        classes_mapping.select_all()
        button_download_data.text = "Change data"
        toggle_cards(["inference_type_selection_card"], enabled=True)
        if select_florence_model_session.get_selected_id() is not None:
            florence_set_model_type_button.enable()
        if select_sam2_model_session.get_selected_id() is not None:
            sam2_set_model_type_button.enable()
        stepper.set_active_step(2)
        select_models_card.uncollapse()
        button_download_data.enable()
        progress_bar_download_data.hide()
        try:
            select_florence_model_session.set_session_id(get_florence_active_task())
            set_florence_model_type()
        except Exception:
            sly.logger.info("No Florence 2 model sessions found to select automatically.")
        try:
            select_sam2_model_session.set_session_id(get_sam2_active_task())
            set_sam2_model_type()
        except Exception:
            sly.logger.info("No SAM 2.1 model sessions found to select automatically.")


progress_bar_download_data = Progress(hide_on_finish=False)
progress_bar_download_data.hide()
input_project_thmb = ProjectThumbnail()
input_project_thmb.hide()
data_card = Card(
    title="Input Data Selection",
    content=Container(
        [
            dataset_selector,
            progress_bar_download_data,
            input_project_thmb,
            button_download_data,
        ]
    ),
    collapsable=True,
)


# --------------------------------------- Connect To Models -------------------------------------- #


select_florence_model_session = SelectAppSession(team_id=g.team.id, tags=["deployed_florence_2"])
select_sam2_model_session = SelectAppSession(
    team_id=g.team.id, tags=["deployed_nn_object_segmentation"]
)


def get_florence_active_task():
    apps = g.api.app.get_list(
        team_id=g.team.id,
        session_tags=["deployed_florence_2"],
        only_running=True,
        filter=[{"field": "name", "operator": "=", "value": "Serve Florence 2"}],
    )
    for task in apps[0].tasks:
        if task.get("status") == "started":
            return task.get("id")


def get_sam2_active_task():
    apps = g.api.app.get_list(
        team_id=g.team.id,
        session_tags=["deployed_nn_object_segmentation"],
        only_running=True,
        filter=[{"field": "name", "operator": "=", "value": "Serve Segment Anything 2.1"}],
    )
    for task in apps[0].tasks:
        if task.get("status") == "started":
            return task.get("id")


@select_florence_model_session.value_changed
def select_florence_model_session_change(val):
    if val is None:
        florence_set_model_type_button.disable()
    else:
        florence_set_model_type_button.enable()


@select_sam2_model_session.value_changed
def select_sam2_model_session_change(val):
    if val is None:
        sam2_set_model_type_button.disable()
    else:
        sam2_set_model_type_button.enable()


florence_model_info = ModelInfo()
florence_model_set_done = DoneLabel("Model successfully connected.")
florence_model_set_done.hide()
florence_set_model_type_button = Button(text="Select model")
select_florence_model_text = Text(
    f'Select <a href="{g.api.server_address}/ecosystem/apps/serve-florence-2" target="_self">Florence 2</a> model:'
)


sam2_model_info = ModelInfo()
sam2_model_set_done = DoneLabel("Model successfully connected.")
sam2_model_set_done.hide()
sam2_set_model_type_button = Button(text="Select model")
select_sam2_model_text = Text(
    f'Select <a href="{g.api.server_address}/ecosystem/apps/serve-segment-anything-2" target="_self">Segment Anything 2.1</a> model:'
)


@florence_set_model_type_button.click
def set_florence_model_type():
    global F_MODEL_DATA
    if florence_set_model_type_button.text == "Disconnect model":
        F_MODEL_DATA["session_id"] = None
        F_MODEL_DATA["model_meta"] = None
        florence_model_set_done.hide()
        florence_model_info.hide()
        select_florence_model_session.enable()
        florence_set_model_type_button.enable()
        florence_set_model_type_button.text = "Connect to Model"
        toggle_cards(
            ["classess_settings_card", "preview_card", "apply_to_project_card"], enabled=False
        )
        set_classess_prompts_button.disable()
        classes_mapping.disable()
        set_classess_prompts_button.text = "Set settings"
        new_random_images_preview_button.disable()
        get_predictions_preview_button.disable()
        apply_to_project_button.disable()
        stepper.set_active_step(2)
    else:
        model_session_id = select_florence_model_session.get_selected_id()
        if model_session_id is not None:
            try:
                florence_set_model_type_button.disable()
                select_florence_model_session.disable()
                model_meta_json = g.api.task.send_request(
                    model_session_id,
                    "get_output_classes_and_tags",
                    data={},
                )
                sly.logger.info(f"Model meta: {str(model_meta_json)}")
                F_MODEL_DATA["model_meta"] = sly.ProjectMeta.from_json(model_meta_json)
                F_MODEL_DATA["session_id"] = model_session_id
                florence_model_set_done.text = "Model successfully connected."
                florence_model_set_done.show()
                florence_model_info.set_session_id(session_id=model_session_id)
                florence_model_info.show()
                florence_set_model_type_button.text = "Disconnect model"
                florence_set_model_type_button._plain = True
                florence_set_model_type_button.enable()
                toggle_cards(["classess_settings_card"], enabled=True)
                set_classess_prompts_button.enable()
                if sam2_set_model_type_button.text == "Disconnect model":
                    stepper.set_active_step(3)
                    classes_mapping.enable()
                    classess_settings_card.uncollapse()
            except Exception as e:
                sly.app.show_dialog(
                    "Error",
                    f"Cannot to connect to model. Make sure that model is deployed and try again.",
                    status="error",
                )
                sly.logger.warn(f"Cannot to connect to model. {e}")
                florence_set_model_type_button.enable()
                florence_set_model_type_button.text = "Connect to model"
                florence_model_set_done.hide()
                select_florence_model_session.enable()
                florence_model_info.hide()
                toggle_cards(
                    ["classess_settings_card", "preview_card", "apply_to_project_card"],
                    enabled=False,
                )
                set_classess_prompts_button.disable()
                classes_mapping.disable()
                new_random_images_preview_button.disable()
                get_predictions_preview_button.disable()
                apply_to_project_button.disable()
                stepper.set_active_step(2)


@sam2_set_model_type_button.click
def set_sam2_model_type():
    global S_MODEL_DATA
    if sam2_set_model_type_button.text == "Disconnect model":
        S_MODEL_DATA["session_id"] = None
        S_MODEL_DATA["model_meta"] = None
        sam2_model_set_done.hide()
        sam2_model_info.hide()
        select_sam2_model_session.enable()
        sam2_set_model_type_button.enable()
        sam2_set_model_type_button.text = "Connect to model"
        toggle_cards(
            ["classess_settings_card", "preview_card", "apply_to_project_card"], enabled=False
        )
        set_classess_prompts_button.disable()
        set_classess_prompts_button.text = "Set settings"
        classes_mapping.disable()
        new_random_images_preview_button.disable()
        get_predictions_preview_button.disable()
        apply_to_project_button.disable()
        stepper.set_active_step(2)
    else:
        model_session_id = select_sam2_model_session.get_selected_id()
        if model_session_id is not None:
            try:
                sam2_set_model_type_button.disable()
                select_sam2_model_session.disable()
                # get model meta
                model_meta_json = g.api.task.send_request(
                    model_session_id,
                    "get_output_classes_and_tags",
                    data={},
                )
                sly.logger.info(f"Model meta: {str(model_meta_json)}")
                S_MODEL_DATA["model_meta"] = sly.ProjectMeta.from_json(model_meta_json)
                S_MODEL_DATA["session_id"] = model_session_id
                sam2_model_set_done.text = "Model successfully connected."
                sam2_model_set_done.show()
                sam2_model_info.set_session_id(session_id=model_session_id)
                sam2_model_info.show()
                sam2_set_model_type_button.text = "Disconnect model"
                sam2_set_model_type_button._plain = True
                sam2_set_model_type_button.enable()
                toggle_cards(["classess_settings_card"], enabled=True)
                set_classess_prompts_button.enable()
                if florence_set_model_type_button.text == "Disconnect model":
                    stepper.set_active_step(3)
                    classes_mapping.enable()
                    classess_settings_card.uncollapse()
            except Exception as e:
                sly.app.show_dialog(
                    "Error",
                    f"Cannot to connect to model. Make sure that model is deployed and try again.",
                    status="error",
                )
                sly.logger.warn(f"Cannot to connect to model. {e}")
                sam2_set_model_type_button.enable()
                sam2_set_model_type_button.text = "Connect to model"
                sam2_model_set_done.hide()
                select_sam2_model_session.enable()
                sam2_model_info.hide()
                toggle_cards(
                    ["classess_settings_card", "preview_card", "apply_to_project_card"],
                    enabled=False,
                )
                set_classess_prompts_button.disable()
                classes_mapping.disable()
                new_random_images_preview_button.disable()
                get_predictions_preview_button.disable()
                apply_to_project_button.disable()
                stepper.set_active_step(2)


inference_type_florence_container = Container(
    [
        select_florence_model_text,
        select_florence_model_session,
        florence_model_info,
        florence_model_set_done,
        florence_set_model_type_button,
    ]
)
inference_type_sam2_container = Container(
    [
        select_sam2_model_text,
        select_sam2_model_session,
        sam2_model_info,
        sam2_model_set_done,
        sam2_set_model_type_button,
    ]
)

select_models_card = Card(
    title="Connect to Models",
    description="Select served models from list below",
    content=Container(
        [inference_type_florence_container, inference_type_sam2_container], direction="horizontal"
    ),
    collapsable=True,
)
select_models_card.collapse()


# ----------------------------------- Model Input Configuration ---------------------------------- #

set_classess_prompts_button = Button("Set settings")


@set_classess_prompts_button.click
def set_model_input():
    if classess_settings_card.is_disabled() is True:
        set_classess_prompts_button.text = "Set settings"
        toggle_cards(["classess_settings_card"], enabled=True)
        toggle_cards(["preview_card", "apply_to_project_card"], enabled=False)
        new_random_images_preview_button.disable()
        get_predictions_preview_button.disable()
        apply_to_project_button.disable()
        classes_mapping.enable()
        stepper.set_active_step(3)
        classes_mapping.set(g.project_meta.obj_classes)
        classes_mapping.select_all()
        apply_to_project_card.collapse()
        preview_card.collapse()
        output_project_thmb.hide()
        classess_settings_card.uncollapse()
    else:
        g.classes_mapping = classes_mapping.get_mapping()
        set_classess_prompts_button.text = "Change Prompts"
        classes_mapping.disable()
        update_images_preview()
        toggle_cards(["classess_settings_card"], enabled=False)
        toggle_cards(["preview_card", "apply_to_project_card"], enabled=True)
        new_random_images_preview_button.enable()
        get_predictions_preview_button.enable()
        apply_to_project_button.enable()
        stepper.set_active_step(5)
        preview_card.uncollapse()
        apply_to_project_card.uncollapse()


classes_mapping = ClassesMappingWithPrompts([])
classess_selection_tabs = Container(
    [
        Grid(
            columns=1,
            widgets=[classes_mapping],
        ),
    ]
)


classess_settings_card = Card(
    title="Classess and Prompts for Predictions",
    description="Selected classes and configure text prompts",
    content=Container(
        [
            classess_selection_tabs,
            set_classess_prompts_button,
        ]
    ),
    collapsable=True,
)
classess_settings_card.collapse()


# --------------------------------------- Inference Preview -------------------------------------- #

grid_gallery = GridGallery(
    columns_number=g.COLUMNS_COUNT,
    annotations_opacity=0.5,
    show_opacity_slider=True,
    enable_zoom=False,
    sync_views=False,
    fill_rectangle=False,
    show_preview=True,
)
grid_gallery.hide()

new_random_images_preview_button = Button(
    "New random images", icon="zmdi zmdi-refresh", button_size="mini"
)


@new_random_images_preview_button.click
def update_images_preview():
    get_predictions_preview_button.disable()
    global IMAGES_INFO_LIST

    grid_gallery.clean_up()
    NEW_PREVIEW_IMAGES_INFOS = []
    images_info_count = len(IMAGES_INFO_LIST)
    preview_images_number = preview_images_number_input.get_value()
    preview_images_number = min(
        preview_images_number, images_info_count, preview_images_number_input.max
    )
    preview_images_number_input.value = preview_images_number

    grid_gallery.columns_number = min(preview_images_number, g.COLUMNS_COUNT)
    grid_gallery._update_layout()

    NEW_PREVIEW_IMAGES_INFOS = random.sample(IMAGES_INFO_LIST, preview_images_number)

    for i, image in enumerate(NEW_PREVIEW_IMAGES_INFOS):
        grid_gallery.append(
            title=image.name,
            image_url=image.preview_url,
            column_index=int(i % g.COLUMNS_COUNT),
        )
    global PREVIEW_IMAGES_INFOS
    PREVIEW_IMAGES_INFOS = NEW_PREVIEW_IMAGES_INFOS
    get_predictions_preview_button.enable()

    grid_gallery.show()


get_predictions_preview_button = Button(
    "Get Predictions Preview", icon="zmdi zmdi-labels", button_type="success"
)


@get_predictions_preview_button.click
def update_predictions_preview():

    new_random_images_preview_button.disable()

    annotations_map = {}
    mapping = {key: value["value"] for key, value in g.classes_mapping.items()}
    with preview_progress(
        message="Generating predictions...", total=len(PREVIEW_IMAGES_INFOS)
    ) as pbar:
        for i, image_info in enumerate(PREVIEW_IMAGES_INFOS):
            annotations_map[image_info.id] = {"annotations": [], "image_info": image_info}
            image_info: sly.ImageInfo
            inference_settings = {"mapping": mapping, "mode": "text_prompt"}
            f_ann = g.api.task.send_request(
                F_MODEL_DATA["session_id"],
                "inference_image_id",
                data={"image_id": image_info.id, "settings": inference_settings},
                timeout=500,
            )
            f_annotation = inference_json_anno_preprocessing(f_ann, g.project_meta, suffix="bbox")
            s_labels = []
            for label in f_annotation.labels:
                class_name = label.obj_class.name.rstrip("_bbox")
                rectangle = label.geometry.to_json()
                inference_settings.update(
                    {
                        "input_image_id": image_info.id,
                        "mode": "bbox",
                        "rectangle": rectangle,
                        "bbox_class_name": class_name,
                    }
                )
                s_ann = g.api.task.send_request(
                    S_MODEL_DATA["session_id"],
                    "inference_image_id",
                    data={"image_id": image_info.id, "settings": inference_settings},
                    timeout=500,
                )
                s_labels.extend(s_ann["annotation"]["objects"])
            s_ann["annotation"]["objects"] = s_labels
            s_annotation = inference_json_anno_preprocessing(s_ann, g.project_meta, suffix="mask")
            merged_annotation = f_annotation.add_labels(s_annotation.labels)
            annotations_map[image_info.id]["annotations"].append(merged_annotation)
            sly.logger.info(
                f"{i+1} image processed. {len(PREVIEW_IMAGES_INFOS) - (i+1)} images left."
            )

            pbar.update(1)

    grid_gallery.clean_up()
    for i, (id, info) in enumerate(annotations_map.items()):
        image_info = info["image_info"]
        annotations = info["annotations"]
        for annotation in annotations:
            grid_gallery.append(
                image_url=image_info.preview_url,
                annotation=annotation,
                title=image_info.name,
                column_index=int(i % g.COLUMNS_COUNT),
            )
    new_random_images_preview_button.enable()


preview_images_number_input = InputNumber(value=4, min=1, max=60, step=1)
preview_images_number_field = Field(
    title="Number of images in preview",
    description="Select how many images should be in preview gallery",
    content=Empty(),
)

input_and_new_random_images_preview_flexbox = Flexbox(
    widgets=[preview_images_number_input, new_random_images_preview_button]
)


@preview_images_number_input.value_changed
def preview_images_number_changed(preview_images_number):
    if not grid_gallery._data:
        sly.logger.debug("Preview gallery is empty, nothing to update.")
        return

    update_images_preview()


preview_progress = Progress()

preview_buttons_flexbox = Flexbox(
    widgets=[
        # new_random_images_preview_button,
        get_predictions_preview_button,
    ],
)


preview_card = Card(
    title="Inference Preview",
    description="Model prediction result preview",
    content=Container(
        [
            preview_images_number_field,
            input_and_new_random_images_preview_flexbox,
            preview_buttons_flexbox,
            preview_progress,
            grid_gallery,
        ]
    ),
    collapsable=True,
)
preview_card.collapse()


# -------------------------------------- Applying Model Card ------------------------------------- #

save_bbox_switch = Switch(on_text="Yes", off_text="No")
save_bbox_field = Field(
    save_bbox_switch,
    title="Florence 2 Bounding Boxes",
    description="Save bounding boxes annotations in output project",
)
destination_project = DestinationProject(g.workspace.id, project_type=sly.ProjectType.IMAGES)
destination_project_item = Collapse.Item(
    "destination_project",
    "Destination Project Settings",
    Container([destination_project, save_bbox_field]),
)
destination_project_collapse = Collapse([destination_project_item])
destination_project_collapse.set_active_panel("destination_project")
apply_to_project_button = Button("Apply to Project")


@apply_to_project_button.click
def run_model():
    toggle_cards(
        [
            "data_card",
            "inference_type_selection_card",
            "classess_settings_card",
            "preview_card",
            "apply_to_project_card",
        ],
        enabled=False,
    )
    g.save_bboxes = save_bbox_switch.is_on()
    button_download_data.disable()
    set_classess_prompts_button.disable()
    florence_set_model_type_button.disable()
    sam2_set_model_type_button.disable()
    new_random_images_preview_button.disable()
    get_predictions_preview_button.disable()
    output_project_thmb.hide()
    destination_project_collapse.set_active_panel([])
    global IS_IMAGE_PROMPT, F_MODEL_DATA, S_MODEL_DATA

    def get_inference_settings():
        mapping = {key: value["value"] for key, value in g.classes_mapping.items()}
        inference_settings = {"mapping": mapping, "mode": "text_prompt"}
        return inference_settings

    try:
        output_project_info = apply_to_project_event(
            destination_project, get_inference_settings(), F_MODEL_DATA, S_MODEL_DATA
        )

        output_project_thmb.set(output_project_info)
        output_project_thmb.show()
        sly.logger.info("Project was successfully labeled")
    except Exception as e:
        sly.logger.error(f"Something went wrong. Error: {e}")
    finally:
        toggle_cards(["apply_to_project_card"], enabled=True)
        button_download_data.enable()
        set_classess_prompts_button.enable()
        florence_set_model_type_button.enable()
        sam2_set_model_type_button.enable()
        apply_to_project_button.enable()


output_project_thmb = ProjectThumbnail()
output_project_thmb.hide()
apply_to_project_card = Card(
    title="Apply to Project",
    content=Container(
        [
            destination_project_collapse,
            project_progress_bar,
            images_update_bar,
            apply_to_project_button,
            output_project_thmb,
        ]
    ),
    collapsable=True,
)
apply_to_project_card.collapse()


def toggle_cards(cards: List[str], enabled: bool = False):
    global CURRENT_REF_IMAGE_INDEX, REF_IMAGE_HISTORY

    def set_card_state(card, state, elements=[]):
        if state:
            card.enable()
            for element in elements:
                element.enable()
        else:
            card.disable()
            for element in elements:
                element.disable()

    card_mappings = {
        "data_card": (data_card, [dataset_selector]),
        "inference_type_selection_card": (
            select_models_card,
            [select_florence_model_session, select_sam2_model_session],
        ),
        "classess_settings_card": (
            classess_settings_card,
            [classess_selection_tabs],
        ),
        "preview_card": (
            preview_card,
            [grid_gallery, new_random_images_preview_button, get_predictions_preview_button],
        ),
        "apply_to_project_card": (apply_to_project_card, [destination_project]),
    }

    for card in cards:
        if card in card_mappings:
            card_element, elements = card_mappings[card]
            set_card_state(card_element, enabled, [e for e in elements if e is not None])


toggle_cards(
    [
        "inference_type_selection_card",
        "classess_settings_card",
        "preview_card",
        "apply_to_project_card",
    ],
    enabled=False,
)
florence_set_model_type_button.disable()
sam2_set_model_type_button.disable()
set_classess_prompts_button.disable()
apply_to_project_button.disable()

stepper = Stepper(
    widgets=[
        data_card,
        select_models_card,
        classess_settings_card,
        preview_card,
        apply_to_project_card,
    ]
)
app = sly.Application(layout=stepper)
