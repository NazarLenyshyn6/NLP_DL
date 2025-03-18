import os 
from pathlib import Path
from logger import logger


project_template = (f'src/__init__.py', 
                    f'src/params/shapes.py',
                    f'src/params/weights.py',
                    f'src/propagation/forward_propagation.py',
                    f'src/propagation/back_propaagation.py',
                    f'src/prediction.py',
                    f'src/factory/propagation_factory.py',
                    f'src/model/__init__.py',
                    f'src/model/model_builder.py',
                    f'configurations/config_entity.py',
                    f'configurations/config_manager.py',
                    f'configurations/config.yaml',
                    f'utils.py',
                     )

for file_path in project_template:
    file_path = Path(file_path)
    file_dir, _ = os.path.split(file_path)
    
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
        logger.info('Created directory %s', file_dir)
        
    if not os.path.exists(file_path) or not os.path.getsize(file_path):
        with open(file_path, 'w') as f:
            ...
            
        logger.info('% has been created', file_path)
        
    else:
        logger.info('%s exists', file_path)
        
logger.info('Template initialization done')