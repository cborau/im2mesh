Usage
=====
.. contents::
   :depth: 2

Examples of batch usage
------------
You can find simple working examples under the /examples folder.
Parameters used are defined in params.txt for each example.

:caption: Example 1:

::

    python main.py -b --path "./examples/ex1_letters/input"
        --format "image" --n_interp 60  --z_size 90.0
        --target_faces 50000 --sampling_factor 0.1
        --min_mesh_size 5.0 --max_mesh_size 5.0
        --output_dir "./examples/ex1_letters/output" --export_vtk




Modules
------------
.. automodapi:: main
   :no-inheritance-diagram:
   
.. automodapi:: formatreader
   :no-inheritance-diagram:

.. automodapi:: formatwriter
   :no-inheritance-diagram:
   
.. automodapi:: sliceinterpolator
   :no-inheritance-diagram:

.. automodapi:: visualization
   :no-inheritance-diagram: