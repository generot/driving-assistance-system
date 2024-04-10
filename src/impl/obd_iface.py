import can
import time

BUSTYPE = "socketcan"
CHANNEL = "can0"
BITRATE = 500000 #500kbps

POLL_INTERVAL = 10

#Should be a shared variable between processes.
VEHICLE_SPEED = 0

def request_obd_resource(pid_code):
    obd_request_id = 0x7DF
    msg = can.Message(arbitration_id=obd_request_id, is_extended_id=False,
                      data=[2, 0x01, pid_code, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC])

    return msg

def obd_loop(vehicle_speed_shared):
    bus = can.interface.Bus(bustype=BUSTYPE, channel=CHANNEL, bitrate=BITRATE)
    obd_resp_id = 0x7E8

    bus.set_filters(filters=[{ "can_id": obd_resp_id, "can_mask": 0xFFF, "extended": False  }])

    vehicle_speed_pid = 0x0D
    vehicle_speed_req = request_obd_resource(vehicle_speed_pid)

    while True:
        bus.send(vehicle_speed_req)

        resp = bus.recv(POLL_INTERVAL)
        
        if resp == None:
            print("OBD: ECU Timeout, no message received.")
        else:
            vehicle_speed_shared.value = resp.data[3] #First byte of received value

if __name__ == "__main__":
    #main()
    pass
