import json
import time



class R():
    """REUSLT CLASS
    """
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def _set_is_error(status: bool):
        R.is_error = status
    
    @staticmethod
    def _set_is_success(status: bool):
        R.is_success = status
    
    @staticmethod
    def _set_time_stamp():
        R.time_stamp = time.time()
        
    @staticmethod
    def _set_code(code: int):
        R.code = code

    @staticmethod
    def _set_data(data):
        R.data = data

    @staticmethod
    def _set_extra(extra_content):
        R.extra = extra_content
        
    @staticmethod
    def _get_result_dict():
        # R.data的数据类型一定是str，但是为了标准化输出
        # [{"a": b, "c": d}, {"e": f, "g": h}] 和 adjiojweod
        # 我们需要区分对待
        try:
            # 应对[{"a": b, "c": d}, {"e": f, "g": h}]
            # data_list = json.dumps(R.data)
            data_list = R.data

        except json.JSONDecodeError:
            # 如果抛出异常则一定是单纯的字符串，直接将原始值赋值给data_list
            data_list = R.data
        data = {
            "code": R.code,
            "data": data_list,
            "is_error": R.is_error,
            "is_success": R.is_success,
            "extra": R.extra,
            "time_stamp": R.time_stamp
        }
        return data
    
    @staticmethod
    def success(success_content, extra_content = None):
        """RETURN SUCCESS METHOD

        Args:
            success_content (any): the content what you want to return, any type data
            extra_content (any, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        R._set_is_error(False)
        R._set_is_success(True)
        R._set_time_stamp()
        R._set_code(0)
        R._set_data(success_content)
        R._set_extra(extra_content)
        
        data = R._get_result_dict()
        # return json.dumps(data, ensure_ascii=False)
        return data
    
    @staticmethod
    def fail(fail_content = None, extra_content = None):
        R._set_is_error(True)
        R._set_is_success(False)
        R._set_time_stamp()
        R._set_code(1)
        R._set_data(fail_content)
        R._set_extra(extra_content)
        data = R._get_result_dict()
        # return json.dumps(data, ensure_ascii=False)
        return data