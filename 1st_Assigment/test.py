import pyglet
try:
    glu_lib = pyglet.lib.load_library('GLU')
    print("GLU loaded:", glu_lib)
except Exception as e:
    print("Error loading GLU:", e)
