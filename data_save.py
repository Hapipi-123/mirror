import findspark
findspark.init()
#添加此代码
from pyspark import SparkConf, SparkContext

import numpy as np
import pandas as pd 
from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("mirror_corder_day_all") \
    .config("hive.metastore.uris","thrift://hive-meta-marketth.hive.svc.datacloud.17usoft.com:9083") \
    .config("hive.metastore.local", "false") \
    .config("spark.io.compression.codec", "snappy") \
    .config("spark.sql.execution.arrow.enabled", "false") \
    .enableHiveSupport() \
    .getOrCreate()

data_sql = '''

SELECT *
FROM tmp_dm.tmp_ybl_rpt_dc_mirror_pv_day_all_dataset
'''
df_raw = spark .sql(data_sql).toPandas()
df_raw.to_csv('./xwjdata/dataset_uv.csv',index=False)
