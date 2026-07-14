# ROS Adapter Placeholder

The ROS wrapper is intentionally deferred until the target device confirms:

- ROS 1 or ROS 2 distribution;
- left image and camera-info topics;
- radar point-cloud topic and exact field definitions;
- image/radar timestamp synchronization policy;
- left-camera-to-ego and radar-to-ego calibration ownership.

The future node should convert messages into `deploy.input_schema.FrameInput`,
append them to `TemporalFrameBuffer`, and call the shared preprocessor/runner.
ROS message types must not leak into the deployment core modules.
