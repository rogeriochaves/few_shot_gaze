import threading
import os
import time

from AppKit import NSKeyUp, NSEvent, NSBundle
import Quartz
from AppKit import NSSystemDefined

callback = None
last_event = time.time()

# Based on:
# https://gist.github.com/cosven/75523f1e970edcc4da8cf1908a9c1463
# https://stackoverflow.com/a/11004560/996404
def keyboard_tap_callback(proxy, type_, event, refcon):
    global callback, last_event

    NSBundle.mainBundle().infoDictionary()['NSAppTransportSecurity'] =\
        dict(NSAllowsArbitraryLoads=True)
    try:
        key_event = NSEvent.eventWithCGEvent_(event)
    except:
        print("mac event cast error")
        return event

    key_code = key_event.keyCode()
    if key_code == 54:
        now = time.time()
        if now - last_event > 1: # 1 second
            last_event = now
            callback()

    return event


def run_event_loop(callback):
    tap = Quartz.CGEventTapCreate(
        Quartz.kCGSessionEventTap,
        Quartz.kCGHeadInsertEventTap,
        Quartz.kCGEventTapOptionListenOnly,
        Quartz.CGEventMaskBit(Quartz.kCGEventFlagsChanged),
        keyboard_tap_callback,
        None
    )

    run_loop_source = Quartz.CFMachPortCreateRunLoopSource(
        None, tap, 0)
    Quartz.CFRunLoopAddSource(
        Quartz.CFRunLoopGetCurrent(),
        run_loop_source,
        Quartz.kCFRunLoopDefaultMode
    )
    # Enable the tap
    Quartz.CGEventTapEnable(tap, True)
    # and run! This won't return until we exit or are terminated.
    Quartz.CFRunLoopRun()

def init(cb):
    global callback

    callback = cb

    x = threading.Thread(target=run_event_loop, args=(0,))
    x.start()
