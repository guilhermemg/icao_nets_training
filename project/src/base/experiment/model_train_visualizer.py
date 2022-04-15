
import matplotlib.pyplot as plt

from src.base.experiment.benchmark_dataset import BenchmarkDataset

class ModelTrainVisualizer:
    def __init__(self, config_interp):
        self.config_interp = config_interp

    
    def visualize_history(self, history):
        f = None
        if not self.config_interp.is_mtl_model:
            f,ax = plt.subplots(1,2, figsize=(10,5))
            f.suptitle(f'-----{self.config_interp.base_model.name}-----')

            ax[0].plot(history.history['accuracy'])
            ax[0].plot(history.history['val_accuracy'])
            ax[0].set_title('Model Accuracy')
            ax[0].set_ylabel('accuracy')
            ax[0].set_xlabel('epoch')
            ax[0].legend(['train', 'validation'])

            ax[1].plot(history.history['loss'])
            ax[1].plot(history.history['val_loss'])
            ax[1].set_title('Model Loss')
            ax[1].set_ylabel('loss')
            ax[1].set_xlabel('epoch')
            ax[1].legend(['train', 'validation'])

        else:
            f,ax = plt.subplots(2,2, figsize=(20,25))
            f.suptitle(f'-----{self.config_interp.base_model.name}-----')

            if not self.config_interp.use_benchmark_data:
                for _,req in enumerate(self.config_interp.prop_args['icao_data']['reqs']):
                    ax[0][0].plot(history.history[f'{req.value}_accuracy'])
                    ax[0][1].plot(history.history[f'val_{req.value}_accuracy'])

                    ax[1][0].plot(history.history[f'{req.value}_loss'])
                    ax[1][1].plot(history.history[f'val_{req.value}_loss'])
            else:
                if self.config_interp.benchmark_dataset.value['name'] == BenchmarkDataset.MNIST.value['name']:
                    for _,cls in enumerate(BenchmarkDataset.MNIST.value['target_cols']):
                        ax[0][0].plot(history.history[f'{cls}_accuracy'])
                        ax[0][1].plot(history.history[f'val_{cls}_accuracy'])

                        ax[1][0].plot(history.history[f'{cls}_loss'])
                        ax[1][1].plot(history.history[f'val_{cls}_loss'])


            ax[1][0].plot(history.history['loss'], color='red', linewidth=2.0) # total loss

            ax[0][0].set_title('Model Accuracy - Train')
            ax[0][1].set_title('Model Accuracy - Validation')

            ax[0][0].set_ylabel('accuracy')
            ax[0][1].set_ylabel('accuracy')
            ax[0][0].set_xlabel('epoch')
            ax[0][1].set_xlabel('epoch')

            ax[0][0].set_ylim([0,1.1])
            ax[0][1].set_ylim([0,1.1])

            ax[1][0].set_title('Model Loss - Train')
            ax[1][1].set_title('Model Loss - Validation')

            ax[1][0].set_ylabel('loss')
            ax[1][1].set_ylabel('loss')

            ax[1][0].set_xlabel('epoch')
            ax[1][1].set_xlabel('epoch')

            ax[1][0].set_ylim([0,1.5])
            ax[1][1].set_ylim([0,1.5])

            legends = None
            if not self.config_interp.use_benchmark_data:
                legends = [r.value for r in self.config_interp.prop_args['icao_data']['reqs']]
            else:
                if self.config_interp.benchmark_dataset.value['name'] == BenchmarkDataset.MNIST.value['name']:
                    legends = BenchmarkDataset.MNIST.value['target_cols']

            ax[0][0].legend(legends, ncol=4)
            ax[0][1].legend(legends, ncol=4)
            ax[1][0].legend(legends, ncol=4)
            ax[1][1].legend(legends, ncol=4)
        
        plt.show()

        return f