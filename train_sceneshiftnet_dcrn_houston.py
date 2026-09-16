"""Run the existing SceneShiftNet training protocol with a shared DCRN encoder."""
import train_sceneshiftnet_houston as training
from models.sceneshift_net_dcrn import SceneShiftNetDCRN

training.SceneShiftNet = SceneShiftNetDCRN
training.OUT = training.Path(__file__).resolve().parent / 'runs_sceneshiftnet_dcrn/houston'

if __name__ == '__main__':
    training.main()
