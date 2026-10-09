"""Render the coupled OpenFOAM case in ParaView.

    pvpython cases/lvcb-2d/paraview/render_case.py
"""

from pathlib import Path

from paraview.simple import (
    ColorBy,
    GetActiveViewOrCreate,
    GetAnimationScene,
    GetColorTransferFunction,
    GetScalarBar,
    OpenFOAMReader,
    Render,
    ResetCamera,
    SaveScreenshot,
    SaveState,
    Show,
    Text,
)

HERE = Path(__file__).resolve().parent
CASE = HERE.parent / "fluid" / "lvcb.foam"
OUT = HERE / "frames"
OUT.mkdir(parents=True, exist_ok=True)

foam = OpenFOAMReader(FileName=str(CASE), registrationName="lvcb")
foam.MeshRegions = ["internalMesh"]
foam.CellArrays = ["T", "U", "p"]
foam.SkipZeroTime = 0
foam.UpdatePipeline()

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1400, 640]
view.Background = [0.08, 0.09, 0.11]
view.OrientationAxesVisibility = 1
view.CameraParallelProjection = 1

display = Show(foam, view)
display.Representation = "Surface"
ColorBy(display, ("CELLS", "T"))
lut = GetColorTransferFunction("T")
try:
    lut.ApplyPreset("Inferno (matplotlib)", True)
except Exception:
    pass
lut.RescaleTransferFunction(300.0, 20000.0)
bar = GetScalarBar(lut, view) if False else None
display.SetScalarBarVisibility(view, True)
display.RescaleTransferFunctionToDataRange(False, True)

scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(foam.TimestepValues)
print("timesteps", times)

# Look straight at the planar mesh (thickness is z).
def frame_camera():
    ResetCamera(view)
    view.CameraFocalPoint = [0.025, 0.011, 0.00025]
    view.CameraPosition = [0.025, 0.011, 0.08]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.CameraParallelScale = 0.014
    Render(view)


label = Text(registrationName="timeLabel")
label.Text = ""
label_display = Show(label, view)
label_display.WindowLocation = "Upper Left Corner"
label_display.FontSize = 18
label_display.Color = [1.0, 1.0, 1.0]


def hide_bars():
    for name in ("T", "U", "p"):
        try:
            GetScalarBar(GetColorTransferFunction(name), view).Visibility = 0
        except Exception:
            pass


def shot(name, time, field, component=None, vmin=None, vmax=None):
    scene.AnimationTime = time
    foam.UpdatePipeline(time)
    hide_bars()
    if component is None:
        ColorBy(display, ("CELLS", field))
    else:
        ColorBy(display, ("CELLS", field, component))
    transfer = GetColorTransferFunction(field)
    if vmin is not None:
        transfer.RescaleTransferFunction(vmin, vmax)
    else:
        display.RescaleTransferFunctionToDataRange(False, True)
    display.SetScalarBarVisibility(view, True)
    label.Text = f"t = {time * 1e3:.2f} ms"
    frame_camera()
    path = OUT / name
    SaveScreenshot(str(path), view, ImageResolution=[1400, 640])
    print("wrote", path)


def nearest(target):
    return min(times, key=lambda t: abs(t - target))


frame_camera()
for tag, target in (
    ("1p30ms", 1.30e-3), ("1p44ms", 1.44e-3), ("1p52ms", 1.52e-3), ("1p56ms", 1.56e-3),
):
    shot(f"paraview_T_{tag}.png", nearest(target), "T", vmin=300, vmax=20000)
shot("paraview_U_in_fins.png", nearest(1.56e-3), "U", "Magnitude")

state = HERE / "lvcb.pvsm"
SaveState(str(state))
print("state", state)
