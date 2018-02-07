#include <iostream>
#include <dlib/svm.h>
#include <fstream>

using namespace std;
using namespace dlib;


int main()
{
	//defining a 2D matrix
	typedef matrix<double, 128, 1> sample_type;

	//radial basis kernel
	typedef radial_basis_kernel<sample_type> kernel_type;

	// Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;

    // Now let's put some data into our samples
    // Get Data From CSV files
    // Import the first subject CSV file
    ifstream ip("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID37_S03_BA_05_102616_1259_Run1_raw.csv");
    if(!ip.is_open()) std::cout << "ERROR: File Not Open" << '\n';
    
    unsigned short int count=0;
    while(count != 61440){

        double temp = 0;
        sample_type samp;
        
        string row;
        getline(ip,row);
        stringstream rowdata(row);

        for(int cols = 0; cols <128; ++cols){
            
            string col_data;
            getline(rowdata,col_data,',');
            stringstream col_value(col_data);

            col_value >> samp(cols);

        }

        samples.push_back(samp);
        labels.push_back(+1);
        count += 1;
    }
    ip.close();

    cout << "For the first file, samples.size(): "<< samples.size() << endl;


    //import the 2nd csv file and add the contents to the same matrix
    ifstream ip2("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID38_S03_BA_05_102716_1104_Run1_raw.csv");
    if(!ip2.is_open()) std::cout << "ERROR: File Not Open" << '\n';
    
    unsigned short int count1=0;
    while(count1 != 61440){

        //double temp = 0;
        sample_type samp1;
        
        string row1;
        getline(ip2,row1);
        stringstream rowdata(row1);

        for(int cols = 0; cols <128; ++cols){
            
            string col_data1;
            getline(rowdata,col_data1,',');
            stringstream col_value(col_data1);

            col_value >> samp1(cols);

        }

        samples.push_back(samp1);
        labels.push_back(-1);
        count1 +=1;
    }
    ip2.close();

    cout << "After adding the second file, samples.size(): "<< samples.size() << endl;

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
    cout << "\n" << endl;


    cout <<"\n Take Test data as input!" << endl;
    cout << "\n creating objects for predicting" << endl;
    std::vector<sample_type> pred_samples;
    std::vector<double> pred_labels;
    std::vector<double> og_labels;

    //define matrix for testing
    // create another object using matrix and push values into it for prediction
    sample_type sample;

    ifstream ip3("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID37_S04_BA_05_110716_1514_Run1_raw.csv");
    if(!ip3.is_open()) std::cout << "ERROR: File Not Open" << '\n';
    
    unsigned short int count2=0;
    while(count2 != 61440){

        string row2;
        getline(ip3,row2);
        stringstream rowdata(row2);

        for(int cols = 0; cols <128; ++cols){
            
            string col_data2;
            getline(rowdata,col_data2,',');
            stringstream col_value(col_data2);

            col_value >> sample(cols);

        }

        pred_samples.push_back(sample);
        og_labels.push_back(+1);
        count2 += 1;
        //pred_labels.push_back(-1);
    }
    ip3.close();

    int predsize ;
    predsize = pred_samples.size();
    cout << "Predicting for pred_samples number:" << endl;
    
    float temp_preds;
    for(int i = 0; i<predsize; i++){
    	temp_preds = learned_function(pred_samples[i]);
    	if( temp_preds > 0){
    		pred_labels.push_back(+1);
    	}
    	else{
    		pred_labels.push_back(-1);
    	}

    }

	cout << "Printing part of the predictions" << endl;    
	for(int j=0; j<5; j++){
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