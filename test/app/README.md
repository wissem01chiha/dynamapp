Test here run starting from a model.json.

Block tests should really only be about json ingestion, no functional testing of blocks.
Exceptions to this now are:
 - DataSource. this testing could mobve to test/library/test_source.py, but not a priority as it would not affect CI execution time.

Model tests should only be intended to test json ingestion, and any pre-processing that happens _before_ the system and context are made ready for simulation by wildcat. Executing simulations should be avoided whenever possible, and testing requiring running simulation should be implemented in wildcat directly, i.e. no json.

The following are development debug tests that will eventually go away:
 - Demo*
 - PallasCatModels
