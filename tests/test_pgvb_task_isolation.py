from scripts.prepare_pgvb_dynamic_tasks import physical_name_key


def test_same_city_name_remains_same_group_across_blocks():
 assert physical_name_key('cambridge_block_9','Downing College')==physical_name_key('cambridge_block_12','downing college')
 assert physical_name_key('cambridge_block_9','Main Road')!=physical_name_key('birmingham_block_9','Main Road')
