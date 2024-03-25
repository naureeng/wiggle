# 🐭 wiggle: analyse mouse steering wheel data 🖱️ <img src="https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA4L2pvYjk1Mi0wNTcteC5qcGc.jpg" width="25%" title="bombcell" alt="bombcell" align="right" vspace = "20">

In the International Brain Laboratory, mice use a steering wheel to move a visual stimulus for sugar reward. We show that mice wiggle the wheel as a strategy. Akin to humans shaking a computer mouse to find the cursor on a screen, mice use movement to build contrast when visual stimulus information is limited. 

**wiggle** provides code to reproduce our results. 

## setup
### <code>iblenv</code> environment
Create and setup the iblenv conda environment, following the <a href="https://github.com/int-brain-lab/iblenv" target="_blank">instructions here</a>.

`cd /path/of/choice
conda create --name iblenv python=3.9 --yes
conda activate iblenv
git clone https://github.com/int-brain-lab/iblapps
pip install --editable iblapps
git clone https://github.com/int-brain-lab/iblenv
cd iblenv`

### install <code>statannotations</code>
Install <code>statannotations</code> to perform statistical analysis in Python.

`pip install statannotations` 

Florian Charlier, Marc Weber, Dariusz Izak, Emerson Harkin, Marcin Magnus, 
Joseph Lalli, Louison Fresnais, Matt Chan, Nikolay Markov, Oren Amsalem, 
Sebastian Proost, Agamemnon Krasoulis, getzze, & Stefan Repplinger. (2022). 
Statannotations (v0.6). Zenodo. https://doi.org/10.5281/zenodo.7213391

## usage
Run the whole pipeline with `python3 main.py` 
## contact
E-mail <naureen.ghani.18@ucl.ac.uk> with any questions.

