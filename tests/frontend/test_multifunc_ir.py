from ptodsl import pto, to_ir_module


def meta_data():
    dtype = pto.float32
    ptr_ty = pto.PtrType(dtype)
    return {"ptr_ty": ptr_ty}


@to_ir_module(meta_data=meta_data)
def single_kernel(arg0: "ptr_ty") -> None:
    pass


@to_ir_module(meta_data=meta_data, module=True)
def build_module():
    @pto.func(kernel="vector")
    def worker(arg0: "ptr_ty") -> None:
        pass

    @pto.func(entry=True)
    def entry(arg0: "ptr_ty") -> None:
        pto.call(worker, arg0)


def test_old_single_function_builder():
    text = str(single_kernel)
    assert "func.func @single_kernel" in text
    assert text.count("func.func @") == 1
    assert "func.call" not in text


def test_new_multi_function_builder():
    text = str(build_module)
    assert "func.func @worker" in text
    assert "pto.kernel_kind = #pto.kernel_kind<vector>" in text
    assert "func.func @entry" in text
    assert "attributes {pto.entry}" in text
    assert "call @worker" in text
