# Documentação de NetTrainer

O script NetTrainer.py é usado para executar experimentos com redes neurais. 
Os dados passados por keyword-arguments (kwargs) são usados para configurar as redes,
os treinamentos e os dados que são documentados em cada experimento realizado.

Você tem opções para realizar ou não os treinamentos e para criar ou não um novo
experimento.

## Variáveis de Entrada

```python
kwargs = { 
    # usar ou não o gerenciador de experimentos Neptune
    'use_neptune': True,   

    # nome do projeto no Neptune
    'neptune_project': cfg.NEPTUNE_PROJ,  

    # parâmetros do experimento
    'exp_params' : {   

        # nome do experimento
        'name': 'train_classifier',   

        # descrição do experimento
        'description': 'Using ADAM optimizer',   

        # tags do experimento
        'tags': ['vgg16', 'balanced datasets', 'adagrad', 'no data augmentation'],  
        
        # arquivos com códigos de origem usando no experimento
        'src_files': ['net_trainer_guilherme.py', '../gen_net_trainer.py', '../data_loader.py', '../evaluation.py']   
    },

    # propriedades do experimento
    'properties': {

        # treinar um novo modelo (True) ou usar o último modelo treinado e salvo localmente (False)
        'train_model': False,

        # salvar modelo treinado localmente e no Neptune
        'save_trained_model': False,

        # fazer subamostragem de dataset de entrada para treinar modelo
        'sample_training_data': True,

        # proporção de subamostragem, usado caso sample_training_data é True
        'sample_prop': 1.0,

        # fazer balanceamento de classes (True) ou não (False). Caso sim o balanceamento ocorre a partir da classe menos numerosa no dataset de treino
        'balance_input_data': True,

        # conjunto de datasets usados para treino. Apenas o split de treino (train) desses datasets são carregados e são tratados como um único dataset de treino ao final
        # opções: [MSU, NUAA, REPLAY_ATTACK]
        'train_datasets': [Dataset.NUAA],

        # conjunto de datasets usados para teste. Apenas o split de teste (test) desses datasets são carregados e são tratados como um único dataset de teste ao final
        # opções: [MSU, NUAA, REPLAY_ATTACK]
        'test_datasets': [Dataset.NUAA]
    },

    # parâmetros para treinamento do modelo
    'net_train_params': {

        # modelo pre-treinado usado de base para transfer learning
        # opções: [VGG16, INCEPTION_V3, MOBILENET_V2, RESNET50_V2, VGG19]
        'base_model': BaseModel.VGG16,

        # tamanho do batch
        'batch_size': 32,

        # quantidade de epochs de treino
        'n_epochs': 5,

        # quantidade de epochs para early stopping
        'early_stopping': 10,

        # fazer shuffle de base de treino (True) ou não (False)
        'shuffle': True,

        # quantidade de dense units para treino de cabeça de modelo
        'dense_units': 128,

        # learning rate inicial
        'learning_rate': 1e-3,

        # optimizador usado
        # opções: [ADAM, ADAGRAD, ADAMAX, SGD, SGD_NESTEROV]
        'optimizer': Optimizer.ADAM,

        # taxa de dropout
        'dropout': 0.3,

        # random seed fixado
        'seed': 42,

        # split para dataset de validação, precisa estar entre 0.0 e 1.0
        'validation_split': 0.15
    }
}
```