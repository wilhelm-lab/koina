# Testing

For testing, we use `pytest`. 
The folder structure in the `test` folder mirrors the folder structure in the `models` folder. 
Each model is covered in their own file in the corresponding sub folder.
This especially applies to all pre- and postprocessing models as well as the ensemble model.

Configuration of the server is done in `server_config.py`. This ensures that other running servers can easily be tested.