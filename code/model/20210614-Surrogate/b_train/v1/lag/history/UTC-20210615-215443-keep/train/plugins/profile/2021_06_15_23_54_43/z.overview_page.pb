?	?'?_[???'?_[??!?'?_[??	s??I?0;@s??I?0;@!s??I?0;@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?'?_[??????}???AF?a????Y,?,?}??rEagerKernelExecute 0*	gffff?v@2t
=Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip@K?P???!?ѝ0??K@)???<?;??1T?C?U,=@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::BatchV2?K6l???!?&r<>OX@)?Xl??ƶ?1?#}??e8@:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle@?5[y????!ޒ??5R@)Yl???گ?1????1@:Preprocessing2?
MIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[1]::TensorSlice@v5y?j??!??5υL,@)v5y?j??1??5υL,@:Preprocessing2?
MIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::TensorSlice@?++MJA??!Lк?[?(@)?++MJA??1Lк?[?(@:Preprocessing2F
Iterator::Model-!?lV??!      Y@)_~?Ɍ?u?1W?CW?C??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism4???????!??????X@)?э???s?1?2?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 27.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t20.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9r??I?0;@I??-?3R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????}???????}???!????}???      ??!       "      ??!       *      ??!       2	F?a????F?a????!F?a????:      ??!       B      ??!       J	,?,?}??,?,?}??!,?,?}??R      ??!       Z	,?,?}??,?,?}??!,?,?}??b      ??!       JCPU_ONLYYr??I?0;@b q??-?3R@Y      Y@q?/3???@"?
host?Your program is HIGHLY input-bound because 27.2% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t20.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.2no:
Refer to the TF2 Profiler FAQ2"CPU: B 