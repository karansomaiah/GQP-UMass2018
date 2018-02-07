#include <iostream>
#include <dlib/svm.h>
#include <fstream>

using namespace std;
using namespace dlib;


int main()
{
	//defining a 2D matrix
	typedef matrix<double, 2, 1> sample_type;

	//radial basis kernel
	typedef radial_basis_kernel<sample_type> kernel_type;

	// Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;

    // Now let's put some data into our samples
    for (int r = -20; r <= 20; ++r)
    {
        for (int c = -20; c <= 20; ++c)
        {
            sample_type samp;
            samp(0) = r;
            samp(1) = c;
            samples.push_back(samp);

            // if this point is less than 10 from the origin
            if (sqrt((double)r*r + c*c) <= 10)
                labels.push_back(+1);
            else
                labels.push_back(-1);

        }
    }

    // normalize the samples
    vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);

    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 

    //randomize the order of the samples with the following function call.
    randomize_samples(samples, labels);

    // find the maximum 'nu' value
    const double max_nu = maximum_nu(labels);

    // here we make an instance of the svm_nu_trainer object that uses our kernel type.
    svm_nu_trainer<kernel_type> trainer;

    // Now we train on the full set of data and obtain the resulting decision function.  We
    // use the value of 0.15625 for nu and gamma.  The decision function will return values
    // >= 0 for samples it predicts are in the +1 class and numbers < 0 for samples it
    // predicts to be in the -1 class.
    trainer.set_kernel(kernel_type(0.15625));
    trainer.set_nu(0.15625);
    typedef decision_function<kernel_type> dec_funct_type;
    typedef normalized_function<dec_funct_type> funct_type;

    cout << "\n Decision Function Initialized, now normalizing and training!" << endl;
    funct_type learned_function;
    learned_function.normalizer = normalizer;  // save normalization information
    learned_function.function = trainer.train(samples, labels); // perform the actual SVM training and save the results

    cout << "\nTraining Done! " << endl;
    
    cout << "\n creating objects for predicting" << endl;
    std::vector<sample_type> pred_samples;
    std::vector<double> pred_labels;
    std::vector<double> og_labels;

    sample_type sample;

    sample(0) = 3.123;
    sample(1) = 2;
    pred_samples.push_back(sample);
    og_labels.push_back(+1);
    
    sample(0) = 3.123;
    sample(1) = 8.3545;
    pred_samples.push_back(sample);
    og_labels.push_back(+1);

    sample(0) = 13.123;
    sample(1) = 9.3545;
    pred_samples.push_back(sample);
    og_labels.push_back(-1);

    sample(0) = 13.123;
    sample(1) = 0;
    pred_samples.push_back(sample);
    og_labels.push_back(-1);

    sample(0) = 3.123;
    sample(1) = 2;
    pred_samples.push_back(sample);
    og_labels.push_back(+1);

    int predsize ;
    predsize = pred_samples.size();
    float temppreds;


    for(int i = 0; i<predsize; i++){
    	//cout << "Predicting for pred_samples number:" << (i+1) << endl;
    	temppreds = learned_function(pred_samples[i]);
    	cout << temppreds << endl;
    	if( temppreds >= 0.0)
    	{
    		pred_labels.push_back(+1);
    		//cout << "YAYYY" << endl;
    	}
    	else
    	{
    		pred_labels.push_back(-1);
    	}
    }

    cout << "Printing part of the predictions" << endl;    
	for(int j=0; j<pred_labels.size(); j++){
		cout << pred_labels[j]<< endl;
	}

	float correct_classes=0;
	for(int k=0; k<predsize; k++){
		if( pred_labels[k] == og_labels[k])
			correct_classes+=1;
	}

	float classification_accuracy=0;
	classification_accuracy = (correct_classes/predsize)*100;

	cout << "Classification Accuracy: " << classification_accuracy << endl;

    cout << "Trying cross-validaiton at this point!" << endl;
    cout << "--------------------------------------" << endl;
    cout << "     cross validation accuracy: " << cross_validate_trainer(trainer, samples, labels, 3);

}