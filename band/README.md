### Heart rate monitor for Mi Band 6

This is a Python implementation of the [miband-6-heart-rate-monitor](https://github.com/gzalo/miband-6-heart-rate-monitor)
Check the complete repository here [miband-HR-python](https://github.com/danielsousaoliveira/miband-HR-python)

## Requirements


1. Find gatt_linux.py on your computer (Should be on /home/user/.local/lib/python3.8/site-packages/gatt)
2. Make the following changes on the file gatt_linux.py:

```
Class DeviceManager:

    # Add the following function to the Class
    def notification_query(self, function, device):
        GObject.timeout_add(10000, function, device)

Class Device:

    # Change the following function 
    def _connect(self):
        self._connect_retry_attempt += 1
        try:
            self._object.Connect()
            # Only this line is changed, remove the rest. Some bug doesn't allow to connect with Band 7
            if not self.services:
                self.services_resolved()
```
