# syntheticmind.trainer.connectors: Runtime Composition Boundaries

Connectors translate Trainer configuration into connected runtime components. Each connector owns one complete vertical boundary, including validation, attachment order, conflicts, and teardown handoff.

`AcceleratorConnector` is the first reconstructed boundary. It replaces separate topology, plugin, and precision resolvers with one decision over accelerator, devices, nodes, Strategy, precision, cluster environment, checkpoint I/O, and layer synchronization.

Connectors are internal implementation concepts. They do not introduce alternate fit, validate, test, predict, checkpoint, or logging entrypoints.
