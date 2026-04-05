import dearpygui.dearpygui as dpg
import time
import os


_current_texture_tag = None
_current_texture_registry_tag = "texture_registry"
_image_widget_tag = "image_widget"

def log(dpg: dpg, msg: str):
    old = dpg.get_value("log_console") or ""
    dpg.set_value("log_console", old + msg + "\n")
    dpg.set_y_scroll("log_child", 1e9)

# def show_image_in_viewer(dpg: dpg, image_path: str):
#     """Load image -> create texture -> display it in Image Viewer window."""
#     global _current_texture_tag

#     if not image_path or not os.path.isfile(image_path):
#         # _clear_image_viewer()
#         dpg.delete_item("image_container", children_only=True)
#         dpg.add_text(f"Image not found:\n{image_path}", parent="image_container", wrap=480)
#         return

#     try:
#         w, h, c, data = dpg.load_image(image_path)
#         print(w, h, c, data)
#     except Exception as e:
#         # _clear_image_viewer()
#         dpg.delete_item("image_container", children_only=True)
#         dpg.add_text(f"Failed to load image:\n{image_path}\n\n{e}", parent="image_container", wrap=480)
#         return

#     # delete previous texture if any
#     if _current_texture_tag is not None and dpg.does_item_exist(_current_texture_tag):
#         dpg.delete_item(_current_texture_tag)

#     # create a new texture tag (unique)
#     _current_texture_tag = f"result_texture_{int(time.time()*1000)}"
#     print(_current_texture_tag)

#     # register texture
#     dpg.add_static_texture(w, h, data, tag=_current_texture_tag, parent=_current_texture_registry_tag)

#     # update UI
#     dpg.delete_item("image_container", children_only=True)
#     dpg.add_image(_current_texture_tag, parent="image_container")
#     # dpg.add_spacer(height=8, parent="image_container")
#     # dpg.add_text(f"{os.path.basename(image_path)}  ({w} x {h})", parent="image_container")

# def show_image_in_viewer(image_path: str):
#     global _current_texture_tag

#     if not image_path or not os.path.isfile(image_path):
#         if dpg.does_item_exist("image_container"):
#             dpg.delete_item("image_container", children_only=True)
#             dpg.add_text(f"Image not found:\n{image_path}", parent="image_container", wrap=480)
#         return

#     try:
#         w, h, c, data = dpg.load_image(image_path)
#         print("loaded:", w, h, c)
#     except Exception as e:
#         if dpg.does_item_exist("image_container"):
#             dpg.delete_item("image_container", children_only=True)
#             dpg.add_text(f"Failed to load image:\n{image_path}\n\n{e}", parent="image_container", wrap=480)
#         return

#     old_texture_tag = _current_texture_tag
#     new_texture_tag = f"result_texture_{int(time.time() * 1000)}"

#     dpg.add_static_texture(
#         width=w,
#         height=h,
#         default_value=data,
#         tag=new_texture_tag,
#         parent=_current_texture_registry_tag
#     )
#     print("1")
#     if not dpg.does_item_exist(_image_widget_tag):
#         dpg.delete_item("image_container", children_only=True)
#         dpg.add_image(new_texture_tag, parent="image_container", tag=_image_widget_tag)
#     else:
#         dpg.configure_item(_image_widget_tag, texture_tag=new_texture_tag)
#     print("2")
#     _current_texture_tag = new_texture_tag
#     print("3")
#     if old_texture_tag is not None and dpg.does_item_exist(old_texture_tag):
#         dpg.delete_item(old_texture_tag)
#     print("4")


def _delete_texture_next_frame(texture_tag):
    if texture_tag and dpg.does_item_exist(texture_tag):
        dpg.delete_item(texture_tag)


def show_image_in_viewer(image_path: str):
    global _current_texture_tag

    if not image_path or not os.path.isfile(image_path):
        if dpg.does_item_exist("image_container"):
            dpg.delete_item("image_container", children_only=True)
            dpg.add_text(f"Image not found:\n{image_path}", parent="image_container", wrap=480)
        return

    try:
        w, h, c, data = dpg.load_image(image_path)
        print("loaded:", w, h, c)
    except Exception as e:
        if dpg.does_item_exist("image_container"):
            dpg.delete_item("image_container", children_only=True)
            dpg.add_text(f"Failed to load image:\n{image_path}\n\n{e}", parent="image_container", wrap=480)
        return

    old_texture_tag = _current_texture_tag
    new_texture_tag = f"result_texture_{int(time.time() * 1000)}"

    dpg.add_static_texture(
        width=w,
        height=h,
        default_value=data,
        tag=new_texture_tag,
        parent=_current_texture_registry_tag
    )

    # Rebuild the image widget instead of reconfiguring it in-place
    if dpg.does_item_exist("image_container"):
        dpg.delete_item("image_container", children_only=True)
        dpg.add_image(new_texture_tag, parent="image_container", tag=_image_widget_tag)

    _current_texture_tag = new_texture_tag

    # Delete old texture on the NEXT frame, not now
    if old_texture_tag is not None and dpg.does_item_exist(old_texture_tag):
        next_frame = dpg.get_frame_count() + 1
        dpg.set_frame_callback(
            next_frame,
            lambda sender=None, app_data=None, user_data=old_texture_tag: _delete_texture_next_frame(user_data)
        )