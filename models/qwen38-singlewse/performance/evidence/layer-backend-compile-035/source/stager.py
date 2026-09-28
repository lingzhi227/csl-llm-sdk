"""Measure recurrent ports sharing native MLP RX/TX descriptors and callbacks."""
from pathlib import Path
from stage_gdn_columns_compile import build
from stage_gdn_resource_compile import main

if __name__=='__main__':main(lambda:build(fused=True),Path(__file__))
