# Description: This example demonstrates how to stream video and audio to the network using the RTSP server.
#
# Note: You will need an SD card to run this example.
#
# You can run the rtsp server to stream video and audio to the network

from media.vencoder import *
from media.sensor import *
from media.media import *
import time, os
import network
import _thread
import multimedia as mm
from time import *

AP_SSID = "K230D_BALL"
AP_KEY = "12345678"

def start_wifi_ap():
    ap = network.WLAN(network.AP_IF)
    if not ap.active():
        ap.active(True)
    ap.config(ssid=AP_SSID, key=AP_KEY)
    sleep(1)

    ip_address = ap.ifconfig()[0]
    print("wifi ap ssid:", AP_SSID)
    print("wifi ap key:", AP_KEY)
    print("rtsp url: rtsp://%s:8554/test" % ip_address)
    return ap, ip_address

class RtspServer:
    def __init__(self,session_name="test",port=8554,video_type = mm.multi_media_type.media_h264,enable_audio=False):
        self.session_name = session_name # session name
        self.video_type = video_type  # 视频类型264/265
        self.enable_audio = enable_audio # 是否启用音频
        self.port = port   #rtsp 端口号
        self.rtspserver = mm.rtsp_server() # 实例化rtsp server
        self.start_stream = False #是否启动推流线程
        self.runthread_over = False #推流线程是否结束

    def start(self):
        # 初始化推流
        self._init_stream()
        self.rtspserver.rtspserver_init(self.port)
        # 创建session
        self.rtspserver.rtspserver_createsession(self.session_name,self.video_type,self.enable_audio)
        # 启动rtsp server
        self.rtspserver.rtspserver_start()
        self._start_stream()

        # 启动推流线程
        self.start_stream = True
        _thread.start_new_thread(self._do_rtsp_stream,())


    def stop(self):
        if (self.start_stream == False):
            return
        # 等待推流线程退出
        self.start_stream = False
        while not self.runthread_over:
            sleep(0.1)
        self.runthread_over = False

        # 停止推流
        self._stop_stream()
        self.rtspserver.rtspserver_stop()
        #self.rtspserver.rtspserver_destroysession(self.session_name)
        self.rtspserver.rtspserver_deinit()

    def get_rtsp_url(self):
        return self.rtspserver.rtspserver_getrtspurl(self.session_name)

    def _init_stream(self):
        width = 1280
        height = 720
        width = ALIGN_UP(width, 16)
        # 初始化sensor
        self.sensor = Sensor()
        self.sensor.reset()
        self.sensor.set_framesize(width = width, height = height, alignment=12)
        self.sensor.set_pixformat(Sensor.YUV420SP)
        # 实例化video encoder
        self.encoder = Encoder()
        self.encoder.SetOutBufs(8, width, height)
        # 创建编码器
        chnAttr = ChnAttrStr(self.encoder.PAYLOAD_TYPE_H264, self.encoder.H264_PROFILE_MAIN, width, height)
        self.encoder.Create(chnAttr)
        # 绑定camera和venc
        self.link = MediaManager.link(self.sensor.bind_info()['src'], (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, self.encoder.chn))

    def _start_stream(self):
        # 开始编码
        self.encoder.Start()
        # 启动camera
        self.sensor.run()

    def _stop_stream(self):
        # 停止camera
        self.sensor.stop()
        # 接绑定camera和venc
        del self.link
        # 停止编码
        self.encoder.Stop()
        self.encoder.Destroy()

    def _do_rtsp_stream(self):
        try:
            streamData = StreamData()
            while self.start_stream:
                os.exitpoint()
                # 获取一帧码流
                self.encoder.GetStream(streamData)
                # 推流
                for pack_idx in range(0, streamData.pack_cnt):
                    stream_data = bytes(uctypes.bytearray_at(streamData.data[pack_idx], streamData.data_size[pack_idx]))
                    self.rtspserver.rtspserver_sendvideodata(self.session_name,stream_data, streamData.data_size[pack_idx],1000)
                    #print("stream size: ", streamData.data_size[pack_idx], "stream type: ", streamData.stream_type[pack_idx])
                # 释放一帧码流
                self.encoder.ReleaseStream(streamData)

        except BaseException as e:
            import sys
            sys.print_exception(e)
        finally:
            self.runthread_over = True
            # 停止rtsp server
            self.stop()

        self.runthread_over = True

if __name__ == "__main__":
    os.exitpoint(os.EXITPOINT_ENABLE)
    ap = None
    rtspserver = None
    try:
        ap, ip_address = start_wifi_ap()
        # 创建rtsp server对象
        rtspserver = RtspServer()
        # 启动rtsp server
        rtspserver.start()
        # 打印电脑端实际需要打开的rtsp url
        print("rtsp server start: rtsp://%s:8554/test" % ip_address)
        # 持续推流，按Ctrl-C停止
        while rtspserver.start_stream:
            sleep(1)
    except KeyboardInterrupt:
        print("user stop")
    except BaseException as e:
        import sys
        sys.print_exception(e)
    finally:
        if rtspserver is not None:
            rtspserver.stop()
        if ap is not None:
            ap.active(False)
        print("done")
