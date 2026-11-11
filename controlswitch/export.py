"""Export the shareable synthetic judge dashboard. Licensed match data stays local."""
from .dashboard import export_html

def export_demo(out):
    return export_html(out)
