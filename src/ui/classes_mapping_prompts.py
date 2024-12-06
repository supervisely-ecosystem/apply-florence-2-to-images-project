from pathlib import Path
from supervisely.app.widgets import ClassesMapping, NotificationBox
from jinja2 import Environment

from supervisely.app.jinja2 import create_env
import markupsafe


class ClassesMappingWithPrompts(ClassesMapping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prompts = {}
        if kwargs.get("empty_notification") is None:
            empty_notification = NotificationBox(
                title="No classes",
                description="No classes to select for detection",
            )
        self.empty_notification = empty_notification

    def to_html(self):
        current_dir = Path(__file__).parent.absolute()
        jinja2_sly_env: Environment = create_env(current_dir)
        html = jinja2_sly_env.get_template("template.html").render({"widget": self})
        html = self._wrap_loading_html(self.widget_id, html)
        html = self._wrap_disable_html(self.widget_id, html)
        html = self._wrap_hide_html(self.widget_id, html)
        return markupsafe.Markup(html)

    def add_prompt(self, prompt):
        self._prompts.update({prompt["class_name"]: prompt["prompt"]})

    def get_prompts(self):
        return self._prompts

    def clear_prompts(self):
        self._prompts = {}

    def get_selected_classes(self):
        return [obj_class for obj_class in self._classes if obj_class.selected]

    def get_unselected_classes(self):
        return [obj_class for obj_class in self._classes if not obj_class.selected]

    def get_selected_classes_names(self):
        return [obj_class.name for obj_class in self.get_selected_classes()]

    def get_unselected_classes_names(self):
        return [obj_class.name for obj_class in self.get_unselected_classes()]

    def get_selected_classes_names_with_prompts(self):
        return [
            f"{obj_class.name} ({', '.join(self.get_prompts())})"
            for obj_class in self.get_selected_classes()
        ]
