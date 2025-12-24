import dxcam


class ScreenCapture:
    def __init__(self, fps=20, region=None):
        self.camera = dxcam.create(output_color="BGR")
        self.region = region
        self.camera.start(target_fps=fps, region=region)

    def get_frame(self):
        return self.camera.get_latest_frame()
    
    def stop(self):
        self.camera.stop()
        self.camera.release()




