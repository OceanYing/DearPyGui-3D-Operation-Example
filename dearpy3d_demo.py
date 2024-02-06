import dearpygui.dearpygui as dpg
import math
import time


class GUI:
    def __init__(self) -> None:
        size = 2
        self.verticies = [
            [-size, -size, -size],  # 0 near side
            [ size, -size, -size],  # 1
            [-size,  size, -size],  # 2
            [ size,  size, -size],  # 3
            [-size, -size,  size],  # 4 far side
            [ size, -size,  size],  # 5
            [-size,  size,  size],  # 6
            [ size,  size,  size],  # 7
            [-size, -size, -size],  # 8 left side
            [-size,  size, -size],  # 9
            [-size, -size,  size],  # 10
            [-size,  size,  size],  # 11
            [ size, -size, -size],  # 12 right side
            [ size,  size, -size],  # 13
            [ size, -size,  size],  # 14
            [ size,  size,  size],  # 15
            [-size, -size, -size],  # 16 bottom side
            [ size, -size, -size],  # 17
            [-size, -size,  size],  # 18
            [ size, -size,  size],  # 19
            [-size,  size, -size],  # 20 top side
            [ size,  size, -size],  # 21
            [-size,  size,  size],  # 22
            [ size,  size,  size],  # 23
        ]

        self.colors = [
            [255,   0,   0, 150],
            [255, 255,   0, 150],
            [255, 255, 255, 150],
            [255,   0, 255, 150],
            [  0, 255,   0, 150],
            [  0, 255, 255, 150],
            [  0,   0, 255, 150],
            [  0, 125,   0, 150],
            [128,   0,   0, 150],
            [128,  70,   0, 150],
            [128, 255, 255, 150],
            [128,   0, 128, 150]
        ]

        self.move = (0, 0)
        self.drag = (0, 0)
        self.mouse_pos = (0, 0)
        self.moving = False
        self.moving_middle = False

        self.x_rot = 0
        self.y_rot = 0
        self.z_rot = 0

        self.x_trans = 0
        self.y_trans = 0
        self.z_trans = 0

        self.scale = 50
        self.mouse_mode = False

        self.window_W = 1000
        self.window_H = 600

        self.register_dpg()
    

    
    def register_dpg(self):
        dpg.create_context()
        dpg.create_viewport(width=self.window_W, height=self.window_H)
        # dpg.setup_dearpygui()

        ## register window ##
        with dpg.window(tag="_primary_window", width=self.window_W, height=self.window_H):
            # dpg.add_image("_texture")
            
            # with dpg.window(tag="_visual_cube", label="visual cube", width=600, height=600, pos=[600, 0]):
            with dpg.drawlist(width=self.window_W, height=self.window_H):

                with dpg.draw_layer(tag="main pass", depth_clipping=True, perspective_divide=True, cull_mode=dpg.mvCullMode_Back):

                    with dpg.draw_node(tag="cube"):

                        verticies = self.verticies
                        colors = self.colors

                        dpg.draw_triangle(verticies[1],  verticies[2],  verticies[0], color=[0,0,0.0],  fill=colors[0])
                        dpg.draw_triangle(verticies[1],  verticies[3],  verticies[2], color=[0,0,0.0],  fill=colors[1])
                        dpg.draw_triangle(verticies[7],  verticies[5],  verticies[4], color=[0,0,0.0],  fill=colors[2])
                        dpg.draw_triangle(verticies[6],  verticies[7],  verticies[4], color=[0,0,0.0],  fill=colors[3])
                        dpg.draw_triangle(verticies[9],  verticies[10], verticies[8], color=[0,0,0.0],  fill=colors[4])
                        dpg.draw_triangle(verticies[9],  verticies[11], verticies[10], color=[0,0,0.0], fill=colors[5])
                        dpg.draw_triangle(verticies[15], verticies[13], verticies[12], color=[0,0,0.0], fill=colors[6])
                        dpg.draw_triangle(verticies[14], verticies[15], verticies[12], color=[0,0,0.0], fill=colors[7])
                        dpg.draw_triangle(verticies[18], verticies[17], verticies[16], color=[0,0,0.0], fill=colors[8])
                        dpg.draw_triangle(verticies[19], verticies[17], verticies[18], color=[0,0,0.0], fill=colors[9])
                        dpg.draw_triangle(verticies[21], verticies[23], verticies[20], color=[0,0,0.0], fill=colors[10])
                        dpg.draw_triangle(verticies[23], verticies[22], verticies[20], color=[0,0,0.0], fill=colors[11])
            # pass
        dpg.set_primary_window("_primary_window", True)

        def callback_mouse_mode(sender, app_data):
            self.mouse_mode = 1-self.mouse_mode

        def callback_reset(sender, app_data):
            dpg.set_value("_rot_x", 10)
            dpg.set_value("_rot_y", 45)
            dpg.set_value("_rot_z", 0)
            dpg.set_value("_pan_x", 0)
            dpg.set_value("_pan_y", 0)
            dpg.set_value("_scale", 50)

        with dpg.window(label="Control", width=300, height=300, pos=[700, 0]):

            dpg.add_slider_float(label="rot_x", default_value=10, min_value=0, max_value=359, tag="_rot_x")
            dpg.add_slider_float(label="rot_y", default_value=45, min_value=0, max_value=359, tag="_rot_y")
            dpg.add_slider_float(label="rot_z", default_value=0,  min_value=0, max_value=359, tag="_rot_z")
            dpg.add_slider_float(label="pan_x", default_value=0, min_value=-10, max_value=10, tag="_pan_x")
            dpg.add_slider_float(label="pan_y", default_value=0, min_value=-10, max_value=10, tag="_pan_y")
            dpg.add_slider_float(label="scale", default_value=50, min_value=10, max_value=100, tag="_scale")
            dpg.add_button(label="mouse mode", tag="_button_mouse_mode", callback=callback_mouse_mode)
            dpg.add_text('mouse move', tag="_mouse_move")
            dpg.add_text('mouse drag', tag="_mouse_drag")
            dpg.add_text('moving', tag="_moving")
            dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)

        
        ## register camera handler ##
        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.scale = self.scale + 0.1 * app_data
            dpg.set_value("_scale", self.scale)

        def callback_camera_drag_rotate(sender, app_data):
            if (not dpg.is_item_focused("_primary_window")) or (self.mouse_mode == 0):
                return
            # self.cam.orbit(app_data[1], app_data[2])
            dx, dy = (app_data[1], app_data[2])
            dpg.set_value("_mouse_drag", f'Mouse drag: {dx, dy}')
            self.x_rot = (self.x_rot + 0.01 * dy) % 360
            self.y_rot = (self.y_rot + 0.01 * dx) % 360
            dpg.set_value("_rot_x", self.x_rot)
            dpg.set_value("_rot_y", self.y_rot)

        def callback_camera_drag_pan(sender, app_data):
            if (not dpg.is_item_focused("_primary_window")) or (self.mouse_mode == 0):
                return
            dx, dy = (app_data[1], app_data[2])
            dpg.set_value("_mouse_drag", f'Mouse drag: {dx, dy}')
            self.x_trans = (self.x_trans + 0.003 * dy)
            self.y_trans = (self.y_trans - 0.003 * dx)
            dpg.set_value("_pan_x", self.x_trans)
            dpg.set_value("_pan_y", self.y_trans)


        def toggle_moving():
            self.moving = not self.moving
        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle

        def move_handler(sender, pos, user):
            if (not dpg.is_item_focused("_primary_window")) or (self.mouse_mode == 1):
                self.mouse_pos = pos
                return
            if self.moving:     # rotate
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    # self.rotate(dx, dy)
                    self.x_rot = (self.x_rot - 0.15 * dy) % 360
                    self.y_rot = (self.y_rot - 0.15 * dx) % 360
                    dpg.set_value("_rot_x", self.x_rot)
                    dpg.set_value("_rot_y", self.y_rot)
            if self.moving_middle:  # pan
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.x_trans = (self.x_trans - 0.05 * dy)
                    self.y_trans = (self.y_trans + 0.05 * dx)
                    dpg.set_value("_pan_x", self.x_trans)
                    dpg.set_value("_pan_y", self.y_trans)
            self.mouse_pos = pos
            dpg.set_value("_mouse_move", f'Mouse pos: {self.mouse_pos[0], self.mouse_pos[1]}')
            dpg.set_value("_moving", f'dragging state: {self.moving}')

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())


        ## Launch the gui ##
        dpg.setup_dearpygui()
        dpg.show_viewport()

    
    def render(self):
        while dpg.is_dearpygui_running():
            t = time.time()

            dpg.set_clip_space("main pass", 0, 0, 600, 600, -1.0, 1.0)

            # two options may be triggerred: 
                # --- self.mouse_mode == 1 --- # 
                    # ------ move along with mouse ------ #
                # --- self.mouse_mode == 0 --- #
                    # ------ continuous movement ------ #

            self.x_rot = dpg.get_value('_rot_x')  # 10
            self.y_rot = dpg.get_value('_rot_y')  # 45
            self.z_rot = dpg.get_value('_rot_z')  # 0
            self.scale = dpg.get_value('_scale')
            self.x_trans = dpg.get_value('_pan_x')
            self.y_trans = dpg.get_value('_pan_y')

            # view = dpg.create_fps_matrix([0, 0, 50], 0.0, 0.0)
            view = dpg.create_fps_matrix([self.y_trans, self.x_trans, self.scale], 0.0, 0.0)
            proj = dpg.create_perspective_matrix(math.pi*45.0/180.0, 1.0, 0.1, 100)
            model = dpg.create_rotation_matrix(math.pi*self.x_rot/180.0 , [1, 0, 0])*\
                                    dpg.create_rotation_matrix(math.pi*self.y_rot/180.0 , [0, 1, 0])*\
                                    dpg.create_rotation_matrix(math.pi*self.z_rot/180.0 , [0, 0, 1])
            dpg.apply_transform("cube", proj * view * model)
            
            dpg.render_dearpygui_frame()


GUI().render()
dpg.destroy_context()
