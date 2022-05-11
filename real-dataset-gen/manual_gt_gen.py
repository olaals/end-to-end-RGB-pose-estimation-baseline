import numpy as np
import cv2
from se3_helpers import look_at_SE3
import spatialmath as sm
from renderer import render_scene
import time
import matplotlib.pyplot as plt
import os
import itertools

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.uix.widget import Widget

import PIL
from scipy.signal import convolve2d




class PoseInitSidebar(BoxLayout):
    def __init__(self, left_img, right_img, pose_init):
        super(PoseInitSidebar, self).__init__(orientation='vertical', size_hint=(None,1.0), width=300)
        self.left_img = left_img
        self.right_img = right_img
        self.pose_init = pose_init
        self.increment_rot_button = Button(text="Increment rotation", size_hint=(1.0,None), height=30)
        self.increment_rot_button.bind(on_press=self.increment_rot_callback)
        self.decrement_rot_button = Button(text="Decrement rotation", size_hint=(1.0,None), height=30)
        self.decrement_rot_button.bind(on_press=self.decrement_rot_callback)
        self.continue_button = Button(text="Continue to PnP", size_hint=(1.0, None), height=30)
        self.continue_button.bind(on_press=self.continue_button_callback)
        self.add_widget(self.increment_rot_button)
        self.add_widget(self.decrement_rot_button)
        self.add_widget(self.continue_button)
        self.empty_widget = Widget()
        self.add_widget(self.empty_widget)

    def increment_rot_callback(self, instance):
        self.pose_init.increment_rotation()
        rendered_img = self.pose_init.get_rendered_image()
        self.left_img.update(cv2.cvtColor(np.uint8(rendered_img*255), cv2.COLOR_RGB2BGR))

    def decrement_rot_callback(self, instance):
        self.pose_init.decrement_rotation()
        rendered_img = self.pose_init.get_rendered_image()
        self.left_img.update(cv2.cvtColor(np.uint8(rendered_img*255), cv2.COLOR_RGB2BGR))

    def continue_button_callback(self, instance):
        self.remove_widget(self.empty_widget)
        self.remove_widget(self.increment_rot_button)
        self.remove_widget(self.decrement_rot_button)
        self.remove_widget(self.continue_button)
        self.pose_init.select_render_for_pnp()
        self.init_pnp_options()

    def init_pnp_options(self):
        self.add_widget(Label(text="Click select pixel pairs, press enter", size_hint=(1.0,None), height=30))
        self.add_widget(Label(text="when both images contains crosshair", size_hint=(1.0, None), height=30))
        self.select_pair_button = Button(text="Select pixel pair", size_hint=(1.0, None), height=30)
        self.select_pair_button.bind(on_press=self.select_pair_callback)
        self.solve_pnp = Button(text="Solve PnP", size_hint=(1.0,0), height=30)
        self.solve_pnp.bind(on_press=self.solve_pnp_callback)
        self.add_widget(self.select_pair_button)
        self.add_widget(self.solve_pnp)
        self.add_widget(Widget())

    def select_pair_callback(self, instance):
        self.pose_init.select_point_pair()
        real_img, rend_img = self.pose_init.get_imgs_w_markers()
        self.right_img.update(real_img)
        self.left_img.update(rend_img)

    def solve_pnp_callback(self, instane):
        print("Solving pnp")
        img = self.pose_init.solve_pnp()
        self.left_img.update(img)








class PoseInitGUI(App):
    def __init__(self, pose_init):
        super().__init__()
        self.pose_init = pose_init

    def build(self):
        layout = BoxLayout(orientation='horizontal')
        self.left_img = ImageDisplay()
        self.right_img = ImageDisplay()
        self.sidebar = PoseInitSidebar(self.left_img, self.right_img, self.pose_init)
        layout.add_widget(self.sidebar)
        layout.add_widget(self.left_img)
        layout.add_widget(self.right_img)
        real_img = self.pose_init.get_real_image()
        self.right_img.update(real_img)
        rendered_img = self.pose_init.get_rendered_image()
        self.left_img.update(cv2.cvtColor(np.uint8(rendered_img*255), cv2.COLOR_RGB2BGR))
        return layout

class ImageDisplay(Image):
    def __init__(self):
        super(ImageDisplay, self).__init__()
        self.current_frame = None


    def update(self, image):
        frame = image
        self.current_frame = frame
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #image_texture = Texture.create(colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.texture = image_texture


class PoseInit():
    def __init__(self, K, cam_config, real_path, model3d_path, num_options=16, z_incr=0.1):
        self.K = K
        self.cam_config = cam_config
        self.real_path = real_path
        self.model3d_path = model3d_path
        self.num_options = num_options
        self.z_incr = z_incr
        self.current_estimate = look_at_SE3([1.6,0,1.2], [0,0,0.25], [0,0,1]).inv()
        self.real_point_pairs = []
        self.render_point_pairs = []
        self.selected_render_point = None
        self.selected_real_point = None
        self.selected_render = None
        self.selected_render_depth = None

    def get_real_image(self):
        img = cv2.imread(self.real_path)
        return img

    def get_rendered_image(self):
        print(self.model3d_path)
        print(self.current_estimate.data[0])
        print(self.cam_config)
        model3d_path = self.model3d_path
        T_CO = self.current_estimate.data[0]
        cam_conf = self.cam_config
        img, d = render_scene(model3d_path, T_CO, cam_conf)
        
        return img
    
    def increment_rotation(self):
        z_rot = sm.SE3.Rz(10,unit='deg')
        self.current_estimate = self.current_estimate*z_rot

    def decrement_rotation(self):
        z_rot = sm.SE3.Rz(-10,unit='deg')
        self.current_estimate = self.current_estimate*z_rot

    def append_real_point_pair(xy):
        self.real_point_pairs.append(xy)

    def append_render_point_pair(xy):
        self.render_point_pairs.append(xy)

    def is_valid_render_point(self, x,y):
        return abs(self.selected_render_depth[y,x])>0.0

    def click_event_render(self, event, x, y, flags, params):
        if(event == cv2.EVENT_LBUTTONDOWN):
            print(self.selected_render_depth[y,x])
            if self.is_valid_render_point(x,y):
                self.selected_render_point = (x,y)
                img = self.selected_render.copy()
                cv_img = cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2BGR)
                cv_img = cv2.drawMarker(cv_img, (x, y),  (0, 0, 255), cv2.MARKER_CROSS, 10, 1);
                cv2.imshow('image_render', cv_img)

    def click_event_real(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_real_point = (x,y)
            cv_img = self.get_real_image()
            cv_img = cv2.drawMarker(cv_img, (x, y),  (0, 0, 255), cv2.MARKER_CROSS, 10, 1);
            cv2.imshow('image_real', cv_img)

    def color_marker_cycler(self):
        return itertools.cycle([(255,0,0),(255,255,0),(0,255,0),(0,255,255),(0,0,255),(255,0,255)])

    def get_imgs_w_markers(self):
        col_marker_iter = self.color_marker_cycler()
        real_marks = self.real_point_pairs
        rend_marks = self.render_point_pairs
        rend_img = self.selected_render
        real_img = self.get_real_image()
        rend_img = cv2.cvtColor(np.uint8(rend_img*255), cv2.COLOR_RGB2BGR)
        for real_mark, rend_mark, color in zip(real_marks, rend_marks, col_marker_iter):
            rend_img = cv2.drawMarker(rend_img, rend_mark,  color, cv2.MARKER_CROSS, 10, 1);
            real_img = cv2.drawMarker(real_img, real_mark,  color, cv2.MARKER_CROSS, 10, 1);
        return real_img, rend_img






    def select_render_for_pnp(self):
        model3d_path = self.model3d_path
        T_CO = self.current_estimate.data[0]
        cam_conf = self.cam_config
        img, d = render_scene(model3d_path, T_CO, cam_conf)
        self.selected_render = img
        self.selected_render_depth = d

    def select_point_pair(self):
        self.select_image_coord_render()
        self.select_image_coord_real()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if(self.selected_render_point is not None and self.selected_real_point is not None):
            self.real_point_pairs.append(self.selected_real_point)
            self.render_point_pairs.append(self.selected_render_point)
        self.selected_render_point = None
        self.selected_real_point = None

    def select_image_coord_render(self):
        img = self.get_rendered_image()
        cv2.imshow('image_render', cv2.cvtColor(np.uint8(img*255), cv2.COLOR_RGB2BGR))
        cv2.setMouseCallback('image_render', self.click_event_render)


    def select_image_coord_real(self):
        img = self.get_real_image()
        cv2.imshow('image_real', img)
        cv2.setMouseCallback('image_real', self.click_event_real)

    def solve_pnp(self):
        print("solve_pnp")
        depth = self.selected_render_depth
        rend_points = self.render_point_pairs
        real_points = self.real_point_pairs
        K = self.K
        points_3d_cam = self.project_point_pairs(depth, rend_points, K)
        T_WC = self.current_estimate.inv().data[0]

        points_3d_world = self.transform_points(T_WC, points_3d_cam)
        print(points_3d_world.shape)
        print(points_3d_world)
        T_CO = self.current_estimate.data[0]
        T_CO_new = self.solve_pnp_smcv(points_3d_world, np.array(real_points, dtype=np.float64), T_CO, K)
        self.current_estimate = T_CO_new
        rend = self.get_rendered_image()
        real = cv2.cvtColor(self.get_real_image(), cv2.COLOR_BGR2RGB)/255.0
        rend = self.create_silhouette(rend)
        blend = self.blend_imgs(real, rend, 0.5)
        plt.imshow(blend)
        plt.show()
        return cv2.cvtColor(np.uint8(blend*255), cv2.COLOR_RGB2BGR)



    @staticmethod
    def transform_points(T, points):
        R = T[:3,:3]
        t = T[:3,3]
        points_new = (R@points.T).T+t
        return points_new

    @staticmethod
    def solve_pnp_smcv(points_W, pixels, T_CO_current, K):
        assert points_W.shape[1] == 3
        assert K.shape == (3,3)
        assert T_CO_current.shape == (4,4)

        print("solve_pnp_smcv")
        print(type(points_W))
        print(type(pixels))
        print(type(T_CO_current))
        print(type(K))
        print(points_W.shape)
        print(pixels.shape)
        print(points_W)
        print(pixels)

        if len(points_W) < 5:
            print("POINTS_W < 5, returning identity")
            return sm.SE3.Rx(0)


        R_current = T_CO_current[:3,:3]
        t_c = T_CO_current[:3, 3].reshape((3,1))

        r_c,_ = cv2.Rodrigues(R_current)

        _, rodr_CW, transl,_ = cv2.solvePnPRansac(points_W, pixels, K, np.array([]), rvec=r_c, tvec=t_c, reprojectionError=5.0, useExtrinsicGuess=True)
        #_,rodr_CW, transl = cv2.solvePnP(points_W, pixels, K, np.array([]))
        rodr_CW = rodr_CW.transpose()[0]
        #R_CW = R.from_mrp(rodr_CW).as_matrix()
        R_CW,_ = cv2.Rodrigues(rodr_CW)

        SO3_CW = sm.SO3(R_CW, check=True)
        T_CW = sm.SE3.Rt(SO3_CW, transl)
        return T_CW

    @staticmethod
    def create_silhouette(img):
        gray = np.mean(img, axis=2)
        sil = np.where(gray>0, 1.0, 0.0)
        filt = np.ones((3,3))
        res = convolve2d(sil, filt, mode='same')
        res = np.clip(res, 0.0, 1.0)
        sil = res-sil
        img[sil>0] = (1.0,0.0, 0.0)
        return img


    @staticmethod
    def blend_imgs(im1, im2, alpha):
        im1 = PIL.Image.fromarray(np.uint8(im1*255.0))
        im2 = PIL.Image.fromarray(np.uint8(im2*255.0))
        return np.asarray(PIL.Image.blend(im1,im2,alpha))/255.0







    @staticmethod
    def project_point_pairs(depth, points, K):
        """
        points = np.array(points)
        print(points.shape)
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        s = np.linalg.inv(K)@points.T
        """
        points_3d = []
        K_inv = np.linalg.inv(K)
        for x,y in points:
            pix = np.array([x,y,1], dtype=np.float64)
            depth_xy = depth[y,x]
            s = K_inv@pix
            x = s*depth_xy
            points_3d.append(x)
        return np.array(points_3d)



        print(points.shape)











if __name__ == '__main__':
    K = np.array([[336.43 ,0.0, 152.18 ],
                [.0, 335.045, 155.54],
                [.0,.0,1.0]])
    """
    K = np.array([[336.43 ,0.0, 160.0 ],
                [.0, 335.045, 160.0],
                [.0,.0,1.0]])
    """
    print(K)

    cam_config = {
        "K":K,
        "image_resolution":320
    }
    T_CO = sm.SE3.Rx(0)

    #T_CO = sm.SE3.Tz(3).data[0]
    model3d_path = "node_adapter.ply"
    #img, d = render_scene(model3d_path, T_CO.data[0], cam_config)
    #img, d = render_scene(model3d_path, T_CO.data[0], cam_config)
    num_options = 16
    pose_init = PoseInit(K, cam_config, "real.png", model3d_path)
    time.sleep(1) 
    PoseInitGUI(pose_init).run()
