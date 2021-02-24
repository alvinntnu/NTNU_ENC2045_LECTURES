import os

def perform_post_exec_cleanup(input_nb_name,tag_to_del='injected-parameters'):

    import json
    from traitlets.config import Config
    from nbconvert import NotebookExporter
    import nbformat

    output_nb_name = os.path.basename(input_nb_name).replace("-ans","")
    
    c = Config()
    c.TagRemovePreprocessor.enabled=True # to enable the preprocessor
    c.TagRemovePreprocessor.remove_cell_tags = [tag_to_del]
    c.preprocessors = ['TagRemovePreprocessor'] # previously: c.NotebookExporter.preprocessors

    nb_body, resources = NotebookExporter(config=c).from_filename(input_nb_name)
    nbformat.write(nbformat.from_dict(json.loads(nb_body)), output_nb_name, 4)


perform_post_exec_cleanup('../exercise-ans/1-python-basics-ans.ipynb',"remove-input")