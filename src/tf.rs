#[cxx::bridge(namespace = "tflite")]
mod tf {
    extern "C++" {
        include!("tensorflow/lite/interpreter.h");
        include!("tensorflow/lite/model.h");
        include!("tflite-pose/include/coral_ffi.h");

        type Interpreter;
        type FlatBufferModel;

        fn make_model(path: String) -> UniquePtr<Model>;
        fn make_interpreter(model: UniquePtr<Model>) -> UniquePtr<Interpreter>;
        fn make_interpreter_from_path(path: String) -> UniquePtr<Interpreter>;
    }
}
