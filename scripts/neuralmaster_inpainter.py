import modules.scripts
from modules.processing import StableDiffusionProcessingImg2Img
from fastapi import FastAPI, Response, Query, Body
import gradio as gr

from tools import inpaint

#INPAINTING_FILL_ELEMENTS = ['img2img_inpainting_fill', 'replacer_inpainting_fill']
INPAINTING_FILL_ELEMENTS = ['img2img_inpainting_fill']

class Script(modules.scripts.Script):
    def __init__(self):
        pass

    def title(self):
        return "Neural-Master-Inpainter"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        pass

    def before_process(self, p: StableDiffusionProcessingImg2Img, *args):
        self.__init__()
        if NM_INPAINTER_ELEMENT_INDEX is None:
            return
        if not hasattr(p, 'inpainting_fill'):
            return
        if p.inpainting_fill != NM_INPAINTER_ELEMENT_INDEX:
            return
        if not (hasattr(p, "image_mask") and bool(p.image_mask)):
            return

        init_image, inpaint_mask = inpaint.nm_inpaint(p.init_images[0], p.image_mask)
        print(f"NeuralMaster Inpainter: Inpainting done")

        p.init_images[0] = init_image
        p.image_mask = inpaint_mask
        p.inpainting_fill = 1 # original

NM_INPAINTER_ELEMENT_NAME = 'NeuralMaster'
NM_INPAINTER_ELEMENT_INDEX = None

def addIntoMaskedContent(component, **kwargs):  #Здесь добавляется новый элемент в список выбора режима инпаинта
    elem_id = kwargs.get('elem_id', None)
    if elem_id not in INPAINTING_FILL_ELEMENTS:
        return
    newElement = (NM_INPAINTER_ELEMENT_NAME, NM_INPAINTER_ELEMENT_NAME)
    if newElement not in component.choices:
        component.choices.append(newElement)
    global NM_INPAINTER_ELEMENT_INDEX
    NM_INPAINTER_ELEMENT_INDEX = component.choices.index(newElement)

modules.scripts.script_callbacks.on_after_component(addIntoMaskedContent)

def get_props_api(_: gr.Blocks, app: FastAPI):
    @app.get("/neuralmaster_inpainter/get_props")
    async def get_props():
        return {
            "version": "0.9.0",
            "menu_item_name": NM_INPAINTER_ELEMENT_NAME,
            "menu_item_index": NM_INPAINTER_ELEMENT_INDEX,
        }

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(get_props_api)
except:
    pass