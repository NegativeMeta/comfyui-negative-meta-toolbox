"""An uploadable image reference that is genuinely optional in a subgraph."""

import os

import folder_paths
import nodes


class NegativeMeta_OptionalColorImage:
    NONE = "None (use continuation clip)"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [name for name in os.listdir(input_dir)
                 if os.path.isfile(os.path.join(input_dir, name))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required": {"image": ([cls.NONE] + sorted(files),
                                        {"image_upload": True})}}

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "loaded")
    FUNCTION = "load"
    CATEGORY = "NegativeMeta/Color"

    def load(self, image):
        if image == self.NONE:
            return (None, False)
        return (nodes.LoadImage().load_image(image)[0], True)

    @classmethod
    def IS_CHANGED(cls, image):
        return cls.NONE if image == cls.NONE else nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return True if image == cls.NONE else nodes.LoadImage.VALIDATE_INPUTS(image)


class NegativeMeta_ColorReferenceGate:
    """Keep a subgraph IMAGE connection valid when its uploader is bypassed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "loaded")
    FUNCTION = "select"
    CATEGORY = "NegativeMeta/Color"

    def select(self, image=None):
        return (image, image is not None)
