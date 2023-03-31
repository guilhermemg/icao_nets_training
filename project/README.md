# Documentação de NetTrainer

O script **exp_runner.py** é usado para executar experimentos com redes neurais. 
Os dados passados por keyword-arguments (kwargs) são usados para configurar as redes, os treinamentos e os dados que são documentados em cada experimento realizado.

Você tem opções para realizar ou não os treinamentos e para criar ou não um novo experimento.

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
            'dataset': Dataset.MNIST
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

    # parametros usados em neural architecture search
    'nas_params': {
        # qtde de epochs de treino de controller
        'controller_sampling_epochs': 2,

        # qtde de samples por cada epoch do controller
        'samples_per_controller_epochs': 3,

        # qtde de epochs de treino do controller
        'controller_training_epochs': 5,

        # qtde de epochs de treino de cada arquitetura
        'architecture_training_epochs': 2,

        # fator de desconto de rewards 
        'controller_loss_alpha': 0.9
    },

    # parametros usados em treino de rede LSTM de controller
    'controller_params': {
        # qtde de hidden nodes da LSTM
        'controller_lstm_dim': 100,

        # optimizer da LSTM
        'controller_optimizer': 'Adam',

        # learning rate da LSTM
        'controller_learning_rate': 0.01,

        # decay da LSTM
        'controller_decay': 0.1,

        # momentum da LSTM
        'controller_momentum': 0.0,

        # se vai usar o predictor de acurácia da arquitetura proposta
        # ou não na LSTM
        'controller_use_predictor': False
    },

    # parâmetros para treinamento do modelo
    'mlp_params': {
        # tamanho máximo da arquitetura (para o caso de NAS)
        'max_architecture_length': 5,

        # modelo pre-treinado usado de base para transfer learning
        # opções: [VGG16, INCEPTION_V3, MOBILENET_V2, RESNET50_V2, VGG19]
        'mlp_base_model': BaseModel.MOBILENET_V2,

        # quantidade de epochs de treino
        'mlp_n_epochs': 3,

        # tamanho do batch
        'mlp_batch_size': 64,

        # quantidade de epochs para early stopping
        'mlp_early_stopping': 5,

        # optimizador usado
        # opções: [ADAM, ADAGRAD, ADAMAX, SGD, SGD_NESTEROV]
        'mlp_optimizer': Optimizer.ADAM,

        # learning rate inicial 
        'mlp_learning_rate': 1e-2,

        # decay
        'mlp_decay': 0.0,

        # momentum
        'mlp_momentum': 0.0,

        # fator de dropout
        'mlp_dropout': 0.2,

        # loss function de modelo
        'mlp_loss_function': 'categorical_crossentropy',

        # se vai usar one_shot ou não
        'mlp_one_shot': False
    }
}
```