# coding: utf-8
from time import time
from mlcin.prototypes.renn import RENN
from runner import Runner


class RunnerRPS( Runner ):

    def get_prototypes(self, X, y):

        renn = RENN()
        renn.fit(X, y)
        
        return renn.reduce_data(X, y)


if __name__ == '__main__':

    
    modulo = 'regular10'
    
    if( modulo == 'regular10' ):
    
        runner = RunnerRPS( folds=9, normalize=True, prefix='datasets', module=modulo )

        datasets =            ['glass'   , 'image_segmentation', 'ionosphere'   , 'iris'  ]
        datasets = datasets + ['liver'   , 'pendigits'         , 'pima_diabetes', 'sonar' ]
        datasets = datasets + ['spambase', 'vehicle'           , 'vowel'        , 'wine'  , 'yeast' ]
        
    elif( modulo == 'imbalanced' ):
    
        runner = RunnerRPS( folds=5, normalize=True, prefix='datasets', module=modulo )
    
        datasets =            ['glass1', 'ecoli-0_vs_1', 'iris0'           , 'glass0'          ]
        datasets = datasets + ['ecoli1', 'new-thyroid2', 'new-thyroid1'    , 'ecoli2'          ]
        datasets = datasets + ['glass6', 'glass2'      , 'shuttle-c2-vs-c4', 'glass-0-1-6_vs_5']

    
    runner.set_datasets(datasets)

    beginTime = time()
    runner.run()
    endTime = time()
    
    print "\nTempo: ",(endTime - beginTime), " Segundos\n"

    output = 'dataset\tGen. Accuracy\tMaj. Accuracy\tMin. Accuracy\t'
    output = output + 'AUC. Accuracy\tData Reduction\n'
    print output + runner.get_output_buffer()
