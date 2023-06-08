class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ucf101':
            # folder that contains class labels
            root_dir = '/Path/to/UCF-101'

            # Save preprocess data into output_dir
            output_dir = '/path/to/VAR/ucf101'

            return root_dir, output_dir
        elif database == 'hmdb51':
            # folder that contains class labels
            root_dir = 'F:/dataset/leg_vedio/koutu/aldataset'

            output_dir = 'F:/dataset/leg_vedio/koutu/action_data_out'

            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        return '/path/to/Models/c3d-pretrained.pth'