#include <iostream>
#include <dlib/svm_threaded.h>
#include <dlib/svm.h>
#include <fstream>
#include <vector>
#include <dlib/rand.h>

using namespace std;
using namespace dlib;

int main()
{
	typedef matrix<double, 128, 1> sample_type;

	//radial basis kernel
	//typedef radial_basis_kernel<sample_type> kernel_type;

	// Now we make objects to contain our samples and their respective labels.
    std::vector<sample_type> samples;
    std::vector<double> labels;
    std::vector<string> csvNames;

    //string for storing csv names
    csvNames.push_back("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID37_S03_BA_05_102616_1259_Run1_raw.csv");
	csvNames.push_back("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID38_S03_BA_05_102716_1104_Run1_raw.csv");
	csvNames.push_back("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID39_S03_BA_05_102816_0912_Run1_raw.csv");
	csvNames.push_back("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID37_S04_BA_05_110716_1514_Run1_raw.csv");
	csvNames.push_back("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID38_S04_BA_05_110316_1202_Run1_raw.csv");
	csvNames.push_back("/home/krnapanda/dlib-19.9/gqpCode/data/raw_data_3task/ID39_S04_BA_05_110616_1001_Run1_raw.csv");


	
	for(int i = 0; i < 3; i++){
		
		string file_name;
		file_name = csvNames[i];
		//const char * c = csvNames[i].c_str();

		ifstream ip(file_name);
    	if(!ip.is_open()) std::cout << "ERROR: File Not Open" << '\n';
    
    	for(int cnt = 0; cnt < 30720; cnt++){

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
        	if(i == 0){
        		labels.push_back(1);
        	}
        	else if (i == 1){
        		labels.push_back(2);
        	}
        	else{
        		labels.push_back(3);
        	}
    	}
    	ip.close();
    	cout << "\n Training Data Number: " << i << endl;
	}

	cout << "\n Training Data Loaded" << endl;

	cout << "After adding the second file, samples.size(): "<< samples.size() << endl;

    // normalize the samples
    vector_normalizer<sample_type> normalizer;
    // let the normalizer learn the mean and standard deviation of the samples
    normalizer.train(samples);

    for (unsigned long i = 0; i < samples.size(); ++i)
        samples[i] = normalizer(samples[i]); 

    //randomize the order of the samples with the following function call.
    randomize_samples(samples, labels);

    // In this example program we will work with a one_vs_one_trainer object which stores any 
    // kind of trainer that uses our sample_type samples.
    typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;


    // Finally, make the one_vs_one_trainer.
    ovo_trainer trainer;

    // Next, we will make two different binary classification trainer objects.  One
    // which uses kernel ridge regression and RBF kernels and another which uses a
    // support vector machine and polynomial kernels.  The particular details don't matter.
    // The point of this part of the example is that you can use any kind of trainer object
    // with the one_vs_one_trainer.
    typedef polynomial_kernel<sample_type> poly_kernel;
    typedef radial_basis_kernel<sample_type> rbf_kernel;

    // make the binary trainers and set some parameters
    krr_trainer<rbf_kernel> rbf_trainer;
    svm_nu_trainer<poly_kernel> poly_trainer;
    poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
    rbf_trainer.set_kernel(rbf_kernel(0.1));

    // Now tell the one_vs_one_trainer that, by default, it should use the rbf_trainer
    // to solve the individual binary classification subproblems.
    trainer.set_trainer(rbf_trainer);
    // We can also get more specific.  Here we tell the one_vs_one_trainer to use the
    // poly_trainer to solve the class 1 vs class 2 subproblem.  All the others will
    // still be solved with the rbf_trainer.
    trainer.set_trainer(poly_trainer, 1, 2);

    cout << "\n Decision Function Initialized, now training!" << endl;
    one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);

    cout << "\nTraining Done! " << endl;
    cout << "\n" << endl;

    cout <<"\n Take Test data as input!" << endl;
    cout << "\n creating objects for predicting" << endl;
    std::vector<sample_type> pred_samples;
    std::vector<double> pred_labels;
    std::vector<double> og_labels;

    //sample_type sample;
    for(int i = 3; i < 6; i++){
		
		string file_name;
		file_name = csvNames[i];
		//string file_name;
		//file_name = csvNames[i];

		ifstream ip(file_name);
    	if(!ip.is_open()) std::cout << "ERROR: File Not Open" << '\n';
    
    	for(int cnt = 0; cnt < 30720; cnt++){

        	double temp = 0;
        	sample_type sample;
        
        	string row;
        	getline(ip,row);
        	stringstream rowdata(row);

        	for(int cols = 0; cols <128; ++cols){
            
            	string col_data;
            	getline(rowdata,col_data,',');
            	stringstream col_value(col_data);

            	col_value >> sample(cols);

        	}

        	pred_samples.push_back(sample);
        	if(i == 0){
        		og_labels.push_back(1);
        	}
        	else if (i == 1){
        		og_labels.push_back(2);
        	}
        	else{
        		og_labels.push_back(3);
        	}
    	}
    	ip.close();
    	cout << "\n Test Data Number: " << i << endl;
	}


	cout << "test deserialized function: \n" << test_multiclass_decision_function(df, pred_samples, og_labels) << endl;
	/*
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
	*/
}