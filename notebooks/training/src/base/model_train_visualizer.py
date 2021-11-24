
import matplotlib.pyplot as plt

from base.data_processor import BenchmarkDataset

class ModelTrainVisualizer:
    def __init__(self, prop_args, base_model, is_mtl_model):
        self.prop_args = prop_args
        self.base_model = base_model
        self.is_mtl_model = is_mtl_model
        self.use_benchmark_data = self.prop_args['benchmarking']['use_benchmark_data']
        if self.use_benchmark_data:
            self.benchmark_dataset = self.prop_args['benchmarking']['benchmark_dataset']

    
    def visualize_history(self, history):
        f = None
        if not self.is_mtl_model:
            f,ax = plt.subplots(1,2, figsize=(10,5))
            f.suptitle(f'-----{self.base_model.name}-----')

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
            f.suptitle(f'-----{self.base_model.name}-----')

            if not self.use_benchmark_data:
                for _,req in enumerate(self.prop_args['reqs']):
                    ax[0][0].plot(history.history[f'{req.value}_accuracy'])
                    ax[0][1].plot(history.history[f'val_{req.value}_accuracy'])

                    ax[1][0].plot(history.history[f'{req.value}_loss'])
                    ax[1][1].plot(history.history[f'val_{req.value}_loss'])
            else:
                if self.benchmark_dataset.value['name'] == BenchmarkDataset.MNIST.value['name']:
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
            if not self.use_benchmark_data:
                legends = [r.value for r in self.prop_args['reqs']]
            else:
                if self.benchmark_dataset.value['name'] == BenchmarkDataset.MNIST.value['name']:
                    legends = BenchmarkDataset.MNIST.value['target_cols']

            ax[0][0].legend(legends, ncol=4)
            ax[0][1].legend(legends, ncol=4)
            ax[1][0].legend(legends, ncol=4)
            ax[1][1].legend(legends, ncol=4)
        
        plt.show()

        return f