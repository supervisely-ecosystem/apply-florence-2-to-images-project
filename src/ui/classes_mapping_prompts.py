from pathlib import Path
from supervisely.app.widgets import ClassesMapping, NotificationBox
from jinja2 import Environment
from supervisely.app import DataJson

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

    def disable(self):
        # pylint: disable=no-member
        self._disabled = True
        self._select_all_btn.disable()
        self._deselect_all_btn.disable()
        self.empty_notification.disable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()

    def enable(self):
        # pylint: disable=no-member
        self._disabled = False
        self._select_all_btn.enable()
        self._deselect_all_btn.enable()
        self.empty_notification.enable()
        DataJson()[self.widget_id]["disabled"] = self._disabled
        DataJson().send_changes()
