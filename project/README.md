# Documentação de NetTrainer

O script **exp_runner.py** é usado para executar experimentos com redes neurais. 
Os dados passados por keyword-arguments (kwargs) são usados para configurar as redes,
os treinamentos e os dados que são documentados em cada experimento realizado.

Você tem opções para realizar ou não os treinamentos e para criar ou não um novo
experimento.

## Variáveis de Entrada

```python
kwargs = { 
    # usar ou não o gerenciador de experimentos Neptune
    #  o nome do projeto e api_key devem ser colocados no arquivo de configuração config.py
    'use_neptune': True,   

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
        # approach de MLT ou NAS usada, pode ser do tipo MTL_Approach ou NAS_MTLApproach
        'approach': NAS_MTLApproach.APPROACH_1,

        # caso for testar com dados de benchmark (MNIST, CIFAR-10, etc)        
        'benchmarking': {
            'use_benchmark_data': False,
            'benchmark_dataset': BenchmarkDataset.MNIST,
            'tasks': list(MNIST_TASK)
        },


        'icao_data': {
            'icao_gt': {
                # usar dataset de ground truth (True), ao invés de dataset rotulado automaticamente (False)
                'use_gt_data': True,

                # nomes dos datasets ground truths que devem ser usados e a respectiva partição no treino
                #  se apenas um dataset for usado, ele deve ser passado para o campo train_validation_test
                'gt_names': {
                    'train_validation': [],
                    'test': [],
                    'train_validation_test': [GTName.FVC]
                },
            },
            'icao_dl': {
                'use_dl_data': False,
                'tagger_model': None
            },

            # lista de requisitos
            'reqs': list(ICAO_REQ),

            # usar dataset alinhado (True) ou não (False)
            'aligned': True,
        },

        # fazer balanceamento de classes (True) ou não (False). Caso sim o balanceamento ocorre a partir da classe menos numerosa no dataset de treino
        'balance_input_data': True,
        
        # treinar um novo modelo (True) ou usar o último modelo treinado e salvo localmente (False)
        'train_model': False,

        # salvar modelo treinado localmente e no Neptune
        'save_trained_model': False,
        
        # executar ou não neural architecture search
        'exec_nas': False,
        
        # nome do modelo previamente treinado e que deve estar na pasta prev_trained_models
        'orig_model_experiment_id': 'ICAO-265',

        # fazer subamostragem de dataset de entrada para treinar modelo
        'sample_training_data': True,

        # proporção de subamostragem, usado caso sample_training_data é True
        'sample_prop': 1.0
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

        # quantidade de dense units para treino de cabeça de modelo
        'dense_units': 128,

        # learning rate inicial
        'learning_rate': 1e-3,

        # optimizador usado
        # opções: [ADAM, ADAGRAD, ADAMAX, SGD, SGD_NESTEROV]
        'optimizer': Optimizer.ADAM,

        # taxa de dropout
        'dropout': 0.3
    },
    
    # parametros usados em neural architecture search
    'nas_params': {
        # numero maximo de blocos (dense layers) por branch
        'max_blocks_per_branch': 5,  
        
        # numero de epochs de treino para cada trial
        'n_epochs': 2,
        
        # numero de trials
        'n_trials': 2
    }
}
```