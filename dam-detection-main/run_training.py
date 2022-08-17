import dam_model_funcs as models
import perf_analysis_funcs as analyze
 

model_num = 'bin0002' 
models.binary_classifier_net(model_num)
analyze.plot_training(model_num)
analyze.do_perf_analysis(model_num)
