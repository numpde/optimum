	6[y??d??6[y??d??!6[y??d??	?򢞇?<@?򢞇?<@!?򢞇?<@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:6[y??d??????U???A^,?????Y??N#-???rEagerKernelExecute 0*	??v???{@2t
=Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip@{3j?J>??!tGLOjN@)??? %??1??01z<@:Preprocessing2f
/Iterator::Model::MaxIntraOpParallelism::BatchV2???????!?Uޡ?|X@)F'K????1?l?3%6@:Preprocessing2?
MIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[1]::TensorSlice@K?|%???!G?-?$?0@)K?|%???1G?-?$?0@:Preprocessing2?
MIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::TensorSlice@?? ?????!s_??/@)?? ?????1s_??/@:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle@?鲘?|??!???o?R@)I?"i7???1g?v;?-@:Preprocessing2F
Iterator::Model?1<??X??!      Y@)????5>s?1?^7????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism5_%???!??#s?X@)????&?q?1;P??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 29.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t14.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?򢞇?<@IMCW?Q@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????U???????U???!????U???      ??!       "      ??!       *      ??!       2	^,?????^,?????!^,?????:      ??!       B      ??!       J	??N#-?????N#-???!??N#-???R      ??!       Z	??N#-?????N#-???!??N#-???b      ??!       JCPU_ONLYY?򢞇?<@b qMCW?Q@