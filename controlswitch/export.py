"""Export the shareable named external demo. Challenge match data stays local."""
from .dashboard import export_html

def export_demo(out):
    return export_html(out)
