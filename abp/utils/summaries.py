import os
import logging

# import tensorflow as tf

def clear_summary_path(path_to_summary):
    """ Removes the summaries if it exists """
    pass
    # if tf.gfile.Exists(path_to_summary):
    #     #TODO possible deleting files without asking user. Need a force option!  ' _ '
    #     logging.info("Summaries Exists. Deleting the summaries at %s" % path_to_summary)
    #     tf.gfile.DeleteRecursively(path_to_summary)
