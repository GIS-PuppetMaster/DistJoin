import unittest

from DBInteractor.PilotDataInteractor import PilotDataInteractor
from PilotModel import PilotModel
from pilotscope.Anchor.BaseAnchor.BasePushHandler import CardPushHandler
from pilotscope.Common.Util import pilotscope_exit
from pilotscope.Factory.SchedulerFactory import SchedulerFactory
from pilotscope.PilotConfig import PilotConfig, PostgreSQLConfig
from pilotscope.PilotScheduler import PilotScheduler
from pilotscope.PilotTransData import PilotTransData


class MyPilotModel(PilotModel):
    def load_model(self):
        pass

    def save_model(self):
        pass


class MyCardPushHandler(CardPushHandler):
    def __init__(self, model:PilotModel, config: PilotConfig):
        super().__init__(config)
        self.model = model
        self.config = config
        self.data_interactor = PilotDataInteractor(config)


    def acquire_injected_data(self, sql):
        self.data_interactor.pull_subquery_card()
        data: PilotTransData = self.data_interactor.execute(sql)
        subquery_2_card = data.subquery_2_card

        # get new cardinalities for each sub-query
        subquery = subquery_2_card.keys()
        _, preds_unnorm, t_total = self.model.model.predict(subquery)
        new_subquery_2_card = {sq: str(max(0.0, pred - 1)) for sq, pred in zip(subquery, preds_unnorm)}

        # return new cardinalities for each sub-query
        return new_subquery_2_card

class MscnTest(unittest.TestCase):
    def setUp(self):
        self.config: PilotConfig = PostgreSQLConfig()
        # self.config.db = "stats_tiny"

    def test_mscn(self):
        try:
            model_name = "Duet-Multi"
            mscn_pilot_model: PilotModel = MyPilotModel(model_name)
            mscn_pilot_model.load_model()

            scheduler: PilotScheduler = SchedulerFactory.create_scheduler(self.config)

            # register MSCN algorithm for each SQL query.
            handler = MyCardPushHandler(mscn_pilot_model, self.config)
            scheduler.register_custom_handlers([handler])

            # pretraining_event = MscnPretrainingModelEvent(self.config, mscn_pilot_model, "mscn_pretrain_data_table",
            #                                               enable_collection=True, enable_training=True,
            #                                               training_data_file=None)
            # register required data
            scheduler.register_required_data("", pull_execution_time=True)

            # scheduler.register_events([pretraining_event])
            scheduler.init()

            # evaluating algorithm using test set.
            # sqls = load_test_sql(self.config.db)
            # for i, sql in enumerate(sqls):
            #     scheduler.execute(sql)
        finally:
            pilotscope_exit()