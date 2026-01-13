import dxcam


class ScreenCapture:
    def __init__(self, fps=20, region=None, output_color="BGR"):
        self.camera = dxcam.create(output_color=output_color)
        self.region = region
        self.fps = fps
        self.camera.start(target_fps=fps, region=region, video_mode=True)

    def get_frame(self):
        return self.camera.get_latest_frame()
    
    def stop(self):
        try:
            self.camera.stop()
        finally:
            self.camera.release()




