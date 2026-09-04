# syntheticmind.storage: Artifact Storage Boundaries

`storage/` owns filesystem protocol resolution, local and remote path semantics, text and binary resources, artifact staging, copying, removal, metadata queries, and directory operations shared by checkpoint, logger, profiler, and Trainer persistence.

`filesystem.py` is the single current storage boundary. Additional storage modules require a distinct backend lifecycle or artifact contract rather than a desire to shorten this implementation.
