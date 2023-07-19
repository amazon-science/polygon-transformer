# https://huggingface.co/koajoel/PolyFormer
import os
import torch
import numpy as np
from fairseq import utils,tasks
from utils.checkpoint_utils import load_model_ensemble_and_task
from models.polyformer import PolyFormerModel
import cv2

import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.refcoco import RefcocoTask
from models.polyformer import PolyFormerModel
from PIL import Image
from torchvision import transforms
import cv2
import gradio as gr
import math
from io import BytesIO
import base64
import re
from demo import visual_grounding

title = "PolyFormer for Visual Grounding"

description = """<p style='text-align: center'> <a href='https://polyformer.github.io/' target='_blank'>Project Page</a> | <a href='https://arxiv.org/pdf/2302.07387.pdf' target='_blank'>Paper</a> | <a href='https://github.com/amazon-science/polygon-transformer' target='_blank'>Github Repo</a></p>
                 <p style='text-align: left'> Demo of PolyFormer for referring image segmentation and referring expression comprehension. Upload your own image or click any one of the examples, and write a description about a certain object. Then click \"Submit\" and wait for the results.</p>
"""

examples = [['demo/vases.jpg', 'the blue vase on the left'],
            ['demo/dog.jpg', 'the dog wearing glasses'],
            ['demo/bear.jpeg', 'a bear astronaut in the space'],
            ['demo/unicorn.jpeg', 'a unicorn doing computer vision research'],
            ['demo/pig.jpeg', 'a pig robot preparing a delicious meal'],
            ['demo/otta.png', 'a gentleman otter in a 19th century portrait'],
            ['demo/pikachu.jpeg', 'a pikachu fine-dining  with  a view  to  the  Eiffel Tower'],
            ['demo/cabin.jpeg', 'a small cabin on top of a snowy mountain in the style of Disney art station']
            ]
io = gr.Interface(fn=visual_grounding, inputs=[gr.inputs.Image(type='pil'), "textbox"],
                  outputs=[gr.outputs.Image(label="output", type='numpy'), gr.outputs.Image(label="predicted mask", type='numpy')],
                  title=title, description=description, examples=examples,
                  allow_flagging=False, allow_screenshot=False, cache_examples=False)
io.launch(share=True)

