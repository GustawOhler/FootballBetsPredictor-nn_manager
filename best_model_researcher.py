import numpy as np
import pandas as pd

from dataset_manager.class_definitions import DatasetSplit
from dataset_manager.dataset_manager import get_dataset_ready_to_learn
from models import Match, Season, League, MatchResult
from nn_manager.common import get_profits
from nn_manager.k_fold_validator import print_results_to_csv


class BestModelResearcher:
    def __init__(self, nn_model, raw_test_set: pd.DataFrame, ready_test_set):
        self.nn_model = nn_model
        self.raw_test_set = raw_test_set
        self.ready_test_dataset = ready_test_set

    def simple_evaluation(self, testing_set, row_name):
        metrics_to_show = {'profit': 0, 'precision': 0, 'accumulated_profit': 0, 'how_many_bets': 0, 'count': testing_set[1].shape[0]}
        received_metrics = self.nn_model.evaluate(testing_set[0], testing_set[1], batch_size=testing_set[1].shape[0], verbose=0)
        for index, metric_name in enumerate(self.nn_model.metrics_names):
            if metric_name in metrics_to_show:
                metrics_to_show[metric_name] = received_metrics[index]
        return pd.Series(metrics_to_show, name=row_name)

    def get_league_for_each_match(self):
        return Match.select(Match.id, League.country, League.division).join(Season).join(League).where(Match.id << self.raw_test_set["match_id"].tolist())

    def get_year_for_each_match(self):
        return Match.select(Match.id, Match.date.year).where(Match.id << self.raw_test_set["match_id"].tolist())

    def get_bookmaker_profit_margin_for_each_match(self):
        return Match.select(Match.id,
                            (1.0/Match.average_home_odds + 1.0/Match.average_draw_odds + 1.0/Match.average_away_odds - 1.0).alias('profit_margin'))\
            .where(Match.id << self.raw_test_set["match_id"].tolist())

    def get_matches_sorted_chronologically(self):
        return Match.select(Match.id).where(Match.id << self.raw_test_set["match_id"].tolist()).order_by(Match.date.asc())

    def get_total_dataset_results(self):
        return self.simple_evaluation(self.ready_test_dataset, 'RAZEM')

    def get_matches_for_result(self):
        return Match.select(Match.id, Match.full_time_result).where(Match.id << self.raw_test_set["match_id"].tolist())

    def test_profit_for_leagues(self):
        matches_and_leagues = pd.DataFrame(list(self.get_league_for_each_match().dicts()))
        received_stats = pd.DataFrame()
        for league_denotation, matches_for_league in matches_and_leagues.groupby(['country', 'division']):
            data_for_league = self.raw_test_set.loc[self.raw_test_set['match_id'].isin(matches_for_league['id'])]
            received_stats = received_stats.append(self.simple_evaluation(get_dataset_ready_to_learn(data_for_league, DatasetSplit.TEST),
                                                                          league_denotation[0] + str(league_denotation[1])))
        received_stats = received_stats.append(self.get_total_dataset_results())
        received_stats.to_csv(f'final_results/league_comparision.csv', index=True, float_format='%.6f')

    def test_profit_for_years(self):
        matches_and_years = pd.DataFrame(list(self.get_year_for_each_match().dicts()))
        received_stats = pd.DataFrame()
        for year, matches_for_year in matches_and_years.groupby(['`date`)']):
            data_for_league = self.raw_test_set.loc[self.raw_test_set['match_id'].isin(matches_for_year['id'])]
            received_stats = received_stats.append(self.simple_evaluation(get_dataset_ready_to_learn(data_for_league, DatasetSplit.TEST),
                                                                          str(year)))
        received_stats = received_stats.append(self.get_total_dataset_results())
        received_stats.to_csv(f'final_results/years_comparision.csv', index=True, float_format='%.6f')

    def test_profit_for_bookmaker_margin(self):
        matches_and_margins = pd.DataFrame(list(self.get_bookmaker_profit_margin_for_each_match().dicts()))
        received_stats = pd.DataFrame()
        percentile_categories, bins_range = pd.qcut(matches_and_margins['profit_margin'], 10, retbins=True, labels=False)
        for percentile_category, matches_for_category in matches_and_margins.groupby([percentile_categories]):
            data_for_league = self.raw_test_set.loc[self.raw_test_set['match_id'].isin(matches_for_category['id'])]
            received_stats = received_stats.append(self.simple_evaluation(get_dataset_ready_to_learn(data_for_league, DatasetSplit.TEST),
                                                                          str(f'({bins_range[percentile_category]}, {bins_range[percentile_category+1]}]')))
        received_stats = received_stats.append(self.get_total_dataset_results())
        received_stats.to_csv(f'final_results/margins_comparision.csv', index=True, float_format='%.6f')

    def get_profit_chronologically(self):
        ordered_matches = pd.DataFrame(list(self.get_matches_sorted_chronologically().dicts()))
        ordered_data = self.raw_test_set.set_index('match_id', drop=False)
        ordered_data = ordered_data.reindex(ordered_matches['id'])
        ordered_dataset = get_dataset_ready_to_learn(ordered_data, DatasetSplit.TEST)
        y_pred_classes = self.nn_model.predict(ordered_dataset[0]).argmax(axis=-1)
        y_true_classes = ordered_dataset[1][:, 0:4].argmax(axis=-1)
        profits = get_profits(y_pred_classes, y_true_classes, ordered_dataset[1][:, 4:7])
        np.savetxt('final_results/profit_progress.txt', profits, fmt='%.4f')

    def test_profit_for_results(self):
        matches_and_results = pd.DataFrame(list(self.get_matches_for_result().dicts()))
        matches_and_results['full_time_result'] = matches_and_results['full_time_result'].map(lambda x: x.value)
        received_stats = pd.DataFrame()
        for result, matches_for_result in matches_and_results.groupby(['full_time_result']):
            data_for_result = self.raw_test_set.loc[self.raw_test_set['match_id'].isin(matches_for_result['id'])]
            received_stats = received_stats.append(self.simple_evaluation(get_dataset_ready_to_learn(data_for_result, DatasetSplit.TEST),
                                                                          str(result)))
        received_stats = received_stats.append(self.get_total_dataset_results())
        received_stats.to_csv(f'final_results/results_comparision.csv', index=True, float_format='%.6f')


    def perform_full_research(self):
        self.test_profit_for_results()
        self.test_profit_for_leagues()
        self.test_profit_for_years()
        self.test_profit_for_bookmaker_margin()
        self.get_profit_chronologically()

