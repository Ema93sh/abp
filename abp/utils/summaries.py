import os
import logging
import shutil


def clear_summary_path(path_to_summary):
    """ Removes the summaries if it exists """
    pass
    if os.path.exists(path_to_summary):
        logging.info("Summaries Exists. Deleting the summaries at %s" % path_to_summary)
        shutil.rmtree(path_to_summary)
