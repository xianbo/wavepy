Example Speckle Analyses 
========================


The analyses of the speckle tracking data is split in three steps:
    
    * Pre-processing:
        Consist of loading and cropping the raw image(s) and doing basic operations like normalization, filtering and subtraction of background.
        
    * Data analyses `per se`:
        Here the physics of the method is used to convert the pre processed data to some meaninfull physical information. It will likely require experimental information like pixel size, photon energy and distance sample to detector. It should not require any information about the sample. 
        
    * Post Processing:
        Perform additional steps to the physical data, like integration, mask, filtering and unwrap. At this point some information about the sample may be required. It must produce graphics and results to be presented to others.
        

The two files below present examples of speckle tracking data analyses. The first is ``speckleAnalyses.py`` and it perfrom basic pre processing (interactive crop), data analyses, and save the result in ``hfd5`` format.

Then ``speckleAnalyses_postProcessing.py`` loads the results obtained with ``speckleAnalyses.py``, and perform operations like undersampling, mask, integration and extra calculations to obtain the final desired result, in this case thisckness of the sample. It also plot the results in a meaninful manner.

Download file with pre-precessing and data analyses: :download:`speckleAnalyses.py
<../../../doc/demo/speckleAnalyses.py>`


Download file with post processing and plot of results: :download:`speckleAnalyses_postProcessing.py
<../../../doc/demo/speckleAnalyses_postProcessing.py>`



Code for Speckle Tracking Analyses.py
=====================================

This section contains the speckleAnalyses script.

Download file: :download:`speckleAnalyses.py
<../../../doc/demo/speckleAnalyses.py>`


.. literalinclude:: ../../../doc/demo/speckleAnalyses.py
    :tab-width: 4
    :linenos:
    :language: guess




Speckle Analyses Post Processing
================================

This section contains the speckleAnalyses_postProcessing script.

Download file: :download:`speckleAnalyses_postProcessing.py
<../../../doc/demo/speckleAnalyses_postProcessing.py>`

.. literalinclude:: ../../../doc/demo/speckleAnalyses_postProcessing.py
    :tab-width: 4
    :linenos:
    :language: guess