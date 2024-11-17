from prism.config import Config, LUNAR_LANDER_CFG

ASYNC_TEST_CONFIG = Config(**LUNAR_LANDER_CFG.__dict__)
ASYNC_TEST_CONFIG.run_through_redis = True
ASYNC_TEST_CONFIG.redis_side = "server"
ASYNC_TEST_CONFIG.redis_port = 6379
ASYNC_TEST_CONFIG.redis_host = "localhost"
ASYNC_TEST_CONFIG.num_processes = 1
ASYNC_TEST_CONFIG.log_to_wandb = False
